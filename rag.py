import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

CHROMA_DIR = "./vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions strictly based on the provided document context.

Context:
{context}

Question: {question}

Instructions:
- Answer only from the context above.
- If the answer is not in the context, say "I couldn't find that in the document."
- Be concise and factual. Cite relevant sections when helpful.

Answer:""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_and_index_pdf(pdf_path: str):
    """Load a PDF, split into chunks, embed and store in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore, len(chunks)


def build_rag_chain(vectorstore: Chroma):
    """Build a RAG chain using LCEL."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    rag_chain = RunnableParallel(
        answer=(
            RunnablePassthrough()
            | (lambda x: {"context": format_docs(retriever.invoke(x["question"])), "question": x["question"]})
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        ),
        source_documents=(
            lambda x: retriever.invoke(x["question"])
        ),
    )
    return rag_chain


def ask(chain, question: str) -> dict:
    """Run a question and return answer + sources."""
    result = chain.invoke({"question": question})
    answer = result["answer"]

    sources = []
    seen = set()
    for doc in result["source_documents"]:
        page = doc.metadata.get("page", "?")
        key = (page, doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            sources.append({
                "page": page + 1 if isinstance(page, int) else page,
                "snippet": doc.page_content[:200].replace("\n", " "),
            })

    return {"answer": answer, "sources": sources}