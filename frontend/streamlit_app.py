import streamlit as st
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.retreiver import RetrievalPipeline
from config import CHROMA_PATH, CHROMA_COLLECTION_NAME, MODEL_NAME, RANKING_SCORER_CKPT, CACHE_DIR

st.set_page_config(
    page_title="EBM RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Cache the pipeline so Streamlit does not reload models on every interaction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RetrievalPipeline(
        chroma_path=str(CHROMA_PATH),
        collection_name=CHROMA_COLLECTION_NAME,
        model_name=MODEL_NAME,
        checkpoint_path=str(RANKING_SCORER_CKPT),
        device=device,
        enable_cache=True,
        cache_dir=str(CACHE_DIR),
    )


def run_query(pipeline: RetrievalPipeline, query: str, top_k: int, threshold: float, 
              use_llm: bool, enable_gates: bool, verify_chain: bool, detect_conflicts: bool):
    """Execute query with retrieval, ranking, and optional LLM."""
    with st.spinner("🔍 Processing query through multi-level gates..."):
        return pipeline.answer_query(
            user_query=query,
            top_k=top_k,
            threshold=threshold,
            use_llm=use_llm,
            enable_gates=enable_gates,
            verify_chain=verify_chain,
            detect_conflicts=detect_conflicts,
        )


def render_gate_status(result):
    """Render gate status indicators."""
    gate_status = result.get("gate_status", "unknown")
    gate_results = result.get("gate_results", {})
    
    cols = st.columns(3)
    
    # Gate 1: Query Quality
    with cols[0]:
        gate1 = gate_results.get("gate1_query_quality", {})
        passed = gate1.get("passed", False)
        score = gate1.get("quality_score", 0.0)
        
        status_emoji = "✅" if passed else "❌"
        status_color = "green" if passed else "red"
        
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 0.5rem; border: 2px solid {status_color}; background-color: rgba({"0,128,0" if passed else "255,0,0"}, 0.1);'>
            <h4>{status_emoji} Gate 1: Query Quality</h4>
            <p style='font-size: 1.2rem; font-weight: bold;'>{score:.2%}</p>
            <p style='font-size: 0.8rem; color: gray;'>{"PASSED" if passed else "REJECTED"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gate 2: Retrieval Quality
    with cols[1]:
        gate2 = gate_results.get("gate2_retrieval_quality", {})
        passed = gate2.get("passed", False)
        score = gate2.get("quality_score", 0.0)
        
        status_emoji = "✅" if passed else "❌"
        status_color = "green" if passed else "red"
        
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 0.5rem; border: 2px solid {status_color}; background-color: rgba({"0,128,0" if passed else "255,0,0"}, 0.1);'>
            <h4>{status_emoji} Gate 2: Retrieval Quality</h4>
            <p style='font-size: 1.2rem; font-weight: bold;'>{score:.2%}</p>
            <p style='font-size: 0.8rem; color: gray;'>{"PASSED" if passed else "REJECTED"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gate 3: Evidence Consistency
    with cols[2]:
        gate3 = gate_results.get("gate3_evidence_consistency", {})
        passed = gate3.get("passed", False)
        score = gate3.get("consistency_score", 0.0)
        
        status_emoji = "✅" if passed else "❌"
        status_color = "green" if passed else "red"
        
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 0.5rem; border: 2px solid {status_color}; background-color: rgba({"0,128,0" if passed else "255,0,0"}, 0.1);'>
            <h4>{status_emoji} Gate 3: Evidence Consistency</h4>
            <p style='font-size: 1.2rem; font-weight: bold;'>{score:.2%}</p>
            <p style='font-size: 0.8rem; color: gray;'>{"PASSED" if passed else "REJECTED"}</p>
        </div>
        """, unsafe_allow_html=True)

def render_evidence_verification(evidence_chain):
    """Render sentence-by-sentence evidence verification."""
    if not evidence_chain:
        return
    
    st.markdown("### 🔍 Evidence Chain Verification")
    
    verification_rate = evidence_chain.get("verification_rate", 0.0)
    verified_count = evidence_chain.get("verified_sentences", 0)
    total_count = evidence_chain.get("total_sentences", 0)
    
    # Overall verification status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Verification Rate", f"{verification_rate:.0%}")
    with col2:
        st.metric("Verified Sentences", f"{verified_count}/{total_count}")
    with col3:
        status = "✅ HIGH" if verification_rate >= 0.8 else "⚠️ MEDIUM" if verification_rate >= 0.6 else "❌ LOW"
        st.metric("Confidence", status)
    
    if verification_rate < 0.8:
        st.warning(f"⚠️ Warning: Only {verification_rate:.0%} of claims are verified by evidence. Some information may be hallucinated!")
    
    # Sentence-by-sentence breakdown
    sentences = evidence_chain.get("sentences", [])
    if sentences:
        st.markdown("#### Sentence Verification Details")
        for sent_info in sentences:
            verified = sent_info.get("verified", False)
            sentence = sent_info.get("sentence", "")
            confidence = sent_info.get("confidence", 0.0)
            citations = sent_info.get("citations", [])
            
            icon = "✅" if verified else "❌"
            color = "green" if verified else "red"
            
            with st.container():
                st.markdown(f"""
                <div style='padding: 0.5rem; margin: 0.5rem 0; border-left: 4px solid {color};'>
                    {icon} <strong>{sentence}</strong><br>
                    <small style='color: gray;'>Confidence: {confidence:.2%} | Citations: {citations if citations else "None"}</small>
                </div>
                """, unsafe_allow_html=True)

def render_contradictions(contradictions):
    """Render contradiction detection results."""
    if not contradictions or not contradictions.get("has_contradictions"):
        st.success("✅ No contradictions detected in evidence sources")
        return
    
    st.markdown("### ⚠️ Contradictions Detected")
    
    conflicts = contradictions.get("conflicts", [])
    st.warning(f"Found {len(conflicts)} potential conflict(s) between evidence sources:")
    
    for idx, conflict in enumerate(conflicts, 1):
        with st.expander(f"Conflict {idx}: {conflict.get('type', 'Unknown')} - {conflict.get('severity', 'Unknown')} Severity"):
            st.markdown(f"**Topic:** {conflict.get('topic', 'N/A')}")
            st.markdown(f"**Severity:** {conflict.get('severity', 'N/A')}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📄 Source 1:**")
                st.write(conflict.get("passage1", "N/A")[:300] + "...")
                st.caption(f"Passage #{conflict.get('passage1_idx', 'N/A')}")
            
            with col2:
                st.markdown("**📄 Source 2:**")
                st.write(conflict.get("passage2", "N/A")[:300] + "...")
                st.caption(f"Passage #{conflict.get('passage2_idx', 'N/A')}")
            
            if conflict.get("explanation"):
                st.info(f"💡 **Explanation:** {conflict['explanation']}")

def render_rejection(result):
    """Render rejection page with diagnosis."""
    st.error("### ❌ Query Rejected by Energy Gates")
    
    diagnosis = result.get("diagnosis", {})
    rejection_reason = diagnosis.get("detailed_reason", "Unknown reason")
    
    st.markdown(f"**Reason:** {rejection_reason}")
    
    # Show which gate failed
    gate_results = result.get("gate_results", {})
    failed_gates = []
    for gate_name, gate_info in gate_results.items():
        if not gate_info.get("passed", True):
            failed_gates.append(gate_name)
    
    if failed_gates:
        st.markdown(f"**Failed Gates:** {', '.join(failed_gates)}")
    
    # Actionable suggestions
    suggestions = diagnosis.get("actionable_suggestions", [])
    if suggestions:
        st.markdown("### 💡 How to Fix")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    
    # Energy saved
    energy_saved = result.get("energy_saved", "Unknown")
    st.info(f"⚡ **Energy Saved:** {energy_saved}")

def render_evidence(evidence_passages):
    """Display retrieved evidence passages with scores and metadata."""
    for i, passage in enumerate(evidence_passages, 1):
        score = passage.get("score", 0)
        text = passage.get("text", "No text available")
        metadata = passage.get("metadata", {})
        
        # Determine confidence badge color
        if score >= 0.7:
            badge = "🟢 High"
        elif score >= 0.5:
            badge = "🟡 Medium"
        else:
            badge = "🔴 Low"
        
        with st.expander(f"📄 Evidence {i} - {badge} Confidence (Score: {score:.3f})"):
            st.markdown(f"**Text:** {text}")
            
            # Show metadata
            if metadata:
                st.json(metadata, expanded=False)

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🏥 Evidence-Based Medicine RAG System</h1>
        <p style='font-size: 1.2rem; color: gray;'>Multi-Level Gating • Hallucination Prevention • Evidence Verification</p>
    </div>
    """, unsafe_allow_html=True)

    # Load pipeline
    try:
        with st.spinner("Loading EBM RAG pipeline..."):
            pipeline = load_pipeline()
        st.success("✅ Pipeline loaded successfully!")
    except Exception as exc:
        st.error(f"❌ Failed to load pipeline: {exc}")
        st.stop()

    # Sidebar settings
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Retrieval settings
        with st.expander("🔍 Retrieval Settings", expanded=True):
            top_k = st.slider("Top-k passages", min_value=5, max_value=50, value=10, step=5)
            threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        # Feature toggles
        with st.expander("🛡️ Advanced Features", expanded=True):
            use_llm = st.checkbox("Generate LLM Answer", value=True)
            enable_gates = st.checkbox("Enable Energy Gates", value=True, 
                                      help="Multi-level quality checks to prevent hallucinations")
            verify_chain = st.checkbox("Verify Evidence Chain", value=False,
                                       help="Check each sentence against evidence (slower)")
            detect_conflicts = st.checkbox("Detect Contradictions", value=True,
                                          help="Find conflicts between evidence sources")
        
        st.divider()
        
        # Cache management
        st.markdown("### 💾 Cache")
        if st.button("🗑️ Clear Cache", use_container_width=True):
            pipeline.clear_cache()
            st.toast("Cache cleared successfully!", icon="✅")
        
        # Cache stats
        if pipeline.cache:
            cache_size = len(pipeline.cache)
            st.metric("Cache Entries", cache_size)
        
        st.divider()
        
        # Example queries
        st.markdown("### 📝 Example Queries")
        example_queries = [
            "What is hypertension?",
            "Treatment for type 2 diabetes",
            "Symptoms of myocardial infarction",
            "What are the side effects of aspirin?",
            "When to use antibiotics?"
        ]
        for query in example_queries:
            if st.button(query, use_container_width=True, key=f"ex_{query}"):
                st.session_state.example_query = query
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "example_query" not in st.session_state:
        st.session_state.example_query = None
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        # Chat interface
        for msg_idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    # Check if rejected
                    if msg.get("gate_status") and "rejected" in msg["gate_status"]:
                        render_rejection(msg)
                    else:
                        # Show gate status if available
                        if msg.get("gate_results"):
                            render_gate_status(msg)
                            st.divider()
                        
                        # Show answer
                        if msg.get("answer"):
                            st.markdown("### 📝 Answer")
                            st.write(msg["answer"])
                            
                            # Show follow-up questions if available
                            if msg.get("follow_up_questions"):
                                st.markdown("### 💡 Suggested Follow-up Questions")
                                st.caption("Click any question to explore further:")
                                
                                # Display as clickable pills in rows
                                for i, question in enumerate(msg["follow_up_questions"]):
                                    if st.button(
                                        f"❓ {question}",
                                        key=f"followup_{msg_idx}_{i}",
                                        use_container_width=True,
                                        type="secondary"
                                    ):
                                        st.session_state.example_query = question
                                        st.rerun()
                        
                        # Show evidence
                        if msg.get("evidence"):
                            st.divider()
                            st.markdown(f"### 📚 Evidence ({len(msg['evidence'])} passages)")
                            render_evidence(msg["evidence"])
                        
                        # Show evidence verification
                        if msg.get("evidence_chain"):
                            st.divider()
                            render_evidence_verification(msg["evidence_chain"])
                        
                        # Show contradictions
                        if msg.get("contradictions"):
                            st.divider()
                            render_contradictions(msg["contradictions"])
                        
                        # Show timing
                        st.caption(f"⏱️ Total time: {msg.get('total_time', 0):.2f}s")
        
        # User input
        user_input = st.chat_input("Ask a medical question...")
        
        # Handle example query
        if st.session_state.example_query:
            user_input = st.session_state.example_query
            st.session_state.example_query = None
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Run query
            result = run_query(
                pipeline, 
                user_input.strip(), 
                top_k, 
                threshold, 
                use_llm,
                enable_gates,
                verify_chain,
                detect_conflicts
            )
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "answer": result.get("answer"),
                "follow_up_questions": result.get("follow_up_questions", []),
                "evidence": result.get("filtered_passages", []),
                "gate_status": result.get("gate_status"),
                "gate_results": result.get("gate_results"),
                "evidence_chain": result.get("evidence_chain"),
                "contradictions": result.get("contradictions"),
                "diagnosis": result.get("diagnosis"),
                "energy_saved": result.get("energy_saved"),
                "total_time": result.get("total_time", 0),
            })
            
            st.rerun()
    
    with tab2:
        st.markdown("## 📊 System Analytics")
        
        if len(st.session_state.messages) > 0:
            # Calculate stats
            total_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
            rejected_queries = len([m for m in st.session_state.messages if m["role"] == "assistant" and "rejected" in m.get("gate_status", "")])
            avg_time = sum([m.get("total_time", 0) for m in st.session_state.messages if m["role"] == "assistant"]) / max(1, len([m for m in st.session_state.messages if m["role"] == "assistant"]))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("Rejected Queries", rejected_queries)
            with col3:
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            # Acceptance rate
            acceptance_rate = (total_queries - rejected_queries) / max(1, total_queries)
            st.progress(acceptance_rate, text=f"Acceptance Rate: {acceptance_rate:.0%}")
            
        else:
            st.info("No queries yet. Start chatting to see analytics!")
    
    with tab3:
        st.markdown("""
        ## 🏥 About This System
        
        This is an **Evidence-Based Medicine (EBM) Retrieval-Augmented Generation (RAG)** system with advanced hallucination prevention.
        
        ### 🎯 Key Features
        
        1. **Multi-Level Energy Gating** ⚡
           - Gate 1: Query quality validation
           - Gate 2: Retrieval quality assessment
           - Gate 3: Evidence consistency verification
        
        2. **Trained Ranking Model** 🧠
           - DistilBERT-based encoder (66M parameters)
           - Trained with MNR loss on medical Q&A
           - 82.6% accuracy with hard negatives
        
        3. **Hallucination Prevention** 🛡️
           - Sentence-by-sentence evidence verification
           - Contradiction detection between sources
           - Confidence scoring with energy metrics
        
        4. **Smart Caching** 💾
           - 1GB disk cache with LRU eviction
           - 7-day TTL for cached results
           - Saves 90%+ compute on repeated queries
        
        ### 📚 Model Architecture
        
        ```
        Query → ChromaDB Retrieval → Ranking Model (Energy-Based) 
           → Gate 1 (Query Quality)
           → Gate 2 (Retrieval Quality)  
           → Gate 3 (Evidence Consistency)
           → LLM Generation
           → Evidence Verification
           → Contradiction Detection
        ```
        
        ### ⚡ Energy Score Guide
        
        - **Energy < 0.3**: 🟢 HIGH confidence (good match)
        - **Energy 0.3-0.5**: 🟡 MEDIUM confidence (acceptable)
        - **Energy > 0.5**: 🔴 LOW confidence (reject or warn)
        
        ### 🔒 Safety Features
        
        - Graceful refusal for out-of-scope queries
        - Actionable feedback when rejecting queries
        - Citation enforcement in LLM responses
        - Conservative thresholds to prevent false information
        
        ---
        
        **Version**: 1.0.0  
        **Model**: best_EBM_scorer.ckpt (Trained Jan 2026)  
        **LLM**: GPT-4o-mini via OpenRouter
        """)


if __name__ == "__main__":
    main()
