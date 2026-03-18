import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from rag import load_and_index_pdf, build_rag_chain, ask

# Load API key from .env at startup
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY not found. Add it to your .env file:\n"
        "  OPENAI_API_KEY=sk-..."
    )

# ── App state ──────────────────────────────────────────────────────────────
current_chain = None
current_pdf_name = None
current_vectorstore = None


def upload_and_index(pdf_file):
    global current_chain, current_pdf_name, current_vectorstore

    if pdf_file is None:
        return (
            "⚠️ Please upload a PDF file.",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    # Release ChromaDB lock before deleting the folder (Windows fix)
    if current_vectorstore is not None:
        try:
            current_vectorstore._client._system.stop()
        except Exception:
            pass
        current_vectorstore = None
        current_chain = None

    if os.path.exists("./vectorstore"):
        try:
            shutil.rmtree("./vectorstore")
        except PermissionError:
            return (
                "❌ Could not clear old vectorstore. Restart the app and try again.",
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

    try:
        vectorstore, n_chunks = load_and_index_pdf(pdf_file)
        current_vectorstore = vectorstore
        current_chain = build_rag_chain(vectorstore)
        current_pdf_name = os.path.basename(pdf_file)
        status = f"✅ **{current_pdf_name}** indexed — {n_chunks} chunks. Ready to chat!"
        return (
            status,
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def answer_question(question: str, history: list):
    global current_chain

    if not question.strip():
        return history, ""

    if current_chain is None:
        history.append({"role": "assistant", "content": "⚠️ Please upload and index a PDF first."})
        return history, ""

    history.append({"role": "user", "content": question})

    try:
        result = ask(current_chain, question)
        answer = result["answer"]

        if result["sources"]:
            answer += "\n\n---\n**Sources:**\n"
            for s in result["sources"]:
                answer += f"\n- **Page {s['page']}**: _{s['snippet']}..._"

        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})

    return history, ""


def clear_chat():
    return [], ""


# ── UI ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="PDF RAG — Ask Your Document") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem">
        <h1 style="font-size:2rem; font-weight:700; margin-bottom:0.25rem">
            📄 PDF RAG — Ask Your Document
        </h1>
        <p style="color:#64748b; font-size:0.95rem">
            Upload any PDF · Indexed into ChromaDB · Ask questions in plain English
        </p>
    </div>
    """)

    with gr.Row():

        # ── Left: Setup ──
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Setup")
            pdf_upload = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            index_btn = gr.Button("🚀 Index PDF", variant="primary")
            status_box = gr.Markdown("_Upload a PDF and click Index to begin._")

        # ── Right: Chat ──
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with your PDF")
            chatbot = gr.Chatbot(label="", height=420)
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask anything about the document...",
                    label="",
                    scale=5,
                    show_label=False,
                    interactive=False,
                )
                ask_btn = gr.Button("Ask ➤", variant="primary", scale=1, interactive=False)
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    # ── Events ──
    index_btn.click(
        fn=upload_and_index,
        inputs=[pdf_upload],
        outputs=[status_box, question_box, ask_btn],
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )

    clear_btn.click(fn=clear_chat, outputs=[chatbot, question_box])


if __name__ == "__main__":
    demo.launch(share=False)
