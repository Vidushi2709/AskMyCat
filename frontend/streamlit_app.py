import streamlit as st
import torch
from backend.retreiver import RetrievalPipeline

st.set_page_config(page_title="EBM Retriever", layout="wide")

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Cache the pipeline so Streamlit does not reload models on every interaction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RetrievalPipeline(
        chroma_path="./collections/ebm",
        collection_name="ebm_passages",
        model_name="distilbert-base-uncased",
        checkpoint_path="ranking_scorer.ckpt",
        device=device,
        enable_cache=True,
    )


def run_query(pipeline: RetrievalPipeline, query: str, top_k: int, threshold: float, use_llm: bool):
    """Execute query with retrieval, ranking, and optional LLM."""
    with st.spinner("Running retrieval and ranking..."):
        return pipeline.answer_query(
            user_query=query,
            top_k=top_k,
            threshold=threshold,
            use_llm=use_llm,
        )


def render_evidence(filtered_passages):
    """Render expandable evidence cards."""
    if not filtered_passages:
        st.write("No high-confidence evidence found.")
        return

    for idx, (passage, meta, score) in enumerate(filtered_passages, start=1):
        with st.expander(f"Evidence {idx} • confidence {score:.3f}"):
            st.write(passage)
            if meta:
                st.caption(meta)


def main():
    st.title("🏥 Evidence-Based Medicine QA")
    st.write("Ask medical questions and explore the retrieval, ranking, and LLM reasoning.")

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(f"Failed to load pipeline: {exc}")
        return

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Top-k passages", min_value=5, max_value=25, value=10, step=1)
        threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        use_llm = st.checkbox("Call LLM", value=True)
        
        if st.button("🗑️ Clear Cache"):
            pipeline.clear_cache()
            st.toast("Cache cleared", icon="✅")
    
    # Initialize conversation history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    st.subheader("💬 Conversation")
    conversation_container = st.container()
    
    with conversation_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:  # assistant
                with st.chat_message("assistant"):
                    st.write(msg["answer"])
                    if msg.get("evidence"):
                        with st.expander(f"Evidence ({len(msg['evidence'])} passages)"):
                            render_evidence(msg["evidence"])
                    st.caption(f"⏱️ {msg.get('total_time', 0):.2f}s")
    
    # User input (chat interface)
    st.divider()
    st.subheader("📝 Ask a Question")
    user_input = st.chat_input("Type your medical question here...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Run query
        result = run_query(pipeline, user_input.strip(), top_k, threshold, use_llm)
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": user_input,
            "answer": result.get("answer", "No answer generated."),
            "evidence": result.get("filtered_passages", []),
            "total_time": result.get("total_time", 0),
        })
        
        # Rerun to display the new message
        st.rerun()
    
    # Optional: Show stats in expander
    with st.expander("📊 Debug Info"):
        st.write(f"Messages in history: {len(st.session_state.messages)}")
        if st.session_state.messages:
            st.write(f"Last query time: {st.session_state.messages[-1].get('total_time', 0):.2f}s")


if __name__ == "__main__":
    main()