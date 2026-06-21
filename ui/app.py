import streamlit as st
import time
from ui.state import init_state
from core.graph import compile_graph
from core.models import ResearchBrief

# Page config
st.set_page_config(
    page_title="Ultimate Deep Researcher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize state
init_state()

# Custom CSS for Premium UI Aesthetics
st.markdown("""
<style>
    /* Gradient Background and Typography */
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
        color: #c9d1d9;
    }
    h1 {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 20px;
    }
    /* Card Styling */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    /* Log Panel Scrollbox */
    .log-box {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        height: 250px;
        overflow-y: scroll;
        font-family: 'Courier New', Courier, monospace;
        font-size: 13px;
        color: #58a6ff;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🧬 Ultimate Deep Researcher")
st.write("Autonomous, self-correcting deep research agent framework using LangGraph.")

# Two columns
col_main, col_side = st.columns([3, 1])

with col_main:
    # Research Input
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔍 Research Scope & Input")
    user_query = st.text_area("What would you like to research?", value=st.session_state.user_query, placeholder="Enter research query details...")
    
    constraints = st.text_input("Constraints / Instructions (comma-separated)", placeholder="e.g. use recent publications, exclude patents")
    
    start_btn = st.button("Start Research Phase", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Graph execution wrapper
    if start_btn and user_query:
        st.session_state.user_query = user_query
        st.session_state.logs = []
        
        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [System] [scoping] Compiling scoping graph...")
        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [ScopingAgent] [clarify_with_user] Analyzing query specificity...")
        
        app = compile_graph()
        
        with st.spinner("Analyzing query..."):
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(app.ainvoke({
                    "user_query": user_query,
                    "topic": user_query[:50]
                }))
                st.session_state.graph_state = result
                
                # Check for token usage metrics to update side panel
                token_usage = result.get("token_usage", {})
                if token_usage:
                    # Update cost stats mock/real counts
                    vertex = token_usage.get("vertex_ai", {"input_tokens": 120, "output_tokens": 80, "calls": 1})
                    freellm = token_usage.get("freellmapi", {"input_tokens": 150, "output_tokens": 50, "calls": 2})
                    st.session_state.cost_stats["vertex_calls"] = vertex.get("calls", 0)
                    st.session_state.cost_stats["vertex_input"] = vertex.get("input_tokens", 0)
                    st.session_state.cost_stats["vertex_output"] = vertex.get("output_tokens", 0)
                    st.session_state.cost_stats["vertex_cost"] = (vertex.get("input_tokens", 0) * 0.0375 / 1000000) + (vertex.get("output_tokens", 0) * 0.15 / 1000000)
                    
                    st.session_state.cost_stats["freellm_calls"] = freellm.get("calls", 0)
                    st.session_state.cost_stats["freellm_input"] = freellm.get("input_tokens", 0)
                    st.session_state.cost_stats["freellm_output"] = freellm.get("output_tokens", 0)
                
                if result.get("clarification_needed"):
                    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [ScopingAgent] [clarify_with_user] Ambiguity detected. Prompting user...")
                else:
                    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [ScopingAgent] [write_research_brief] Generated research brief successfully.")
                    
            except Exception as e:
                st.error(f"Error running scoping agent: {str(e)}")
                st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [System] [error] Scoping failed: {str(e)}")

    # Clarification loops
    if st.session_state.graph_state and st.session_state.graph_state.get("clarification_needed"):
        question = st.session_state.graph_state.get("clarification_question")
        st.warning(f"🤔 Clarification Required: {question}")
        response = st.text_input("Please clarify:", value=st.session_state.clarification_response)
        
        if st.button("Submit Clarification"):
            st.session_state.clarification_response = response
            st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [System] [clarify_with_user] User clarification submitted.")
            st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [ScopingAgent] [write_research_brief] Compiling research brief...")
            
            app = compile_graph()
            with st.spinner("Compiling research brief..."):
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(app.ainvoke({
                        "user_query": st.session_state.user_query,
                        "clarification_question": question,
                        "clarification_response": response,
                        "clarification_needed": False
                    }))
                    st.session_state.graph_state = result
                    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [ScopingAgent] [write_research_brief] Generated research brief successfully.")
                    
                    token_usage = result.get("token_usage", {})
                    if token_usage:
                        vertex = token_usage.get("vertex_ai", {"input_tokens": 180, "output_tokens": 120, "calls": 2})
                        st.session_state.cost_stats["vertex_calls"] = vertex.get("calls", 0)
                        st.session_state.cost_stats["vertex_input"] = vertex.get("input_tokens", 0)
                        st.session_state.cost_stats["vertex_output"] = vertex.get("output_tokens", 0)
                        st.session_state.cost_stats["vertex_cost"] = (vertex.get("input_tokens", 0) * 0.0375 / 1000000) + (vertex.get("output_tokens", 0) * 0.15 / 1000000)
                except Exception as e:
                    st.error(f"Error running brief compiler: {str(e)}")
                    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] [System] [error] Scoping failed: {str(e)}")

    # Progress Log box
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📜 Agent Execution Trace")
    if st.session_state.logs:
        log_content = "\n".join(st.session_state.logs)
        st.markdown(f'<div class="log-box">{log_content}</div>', unsafe_allow_html=True)
    else:
        st.info("Agent execution logs will appear here when research starts.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Output Brief Viewer
    if st.session_state.graph_state and st.session_state.graph_state.get("research_brief"):
        brief: ResearchBrief = st.session_state.graph_state["research_brief"]
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📋 Generated Research Brief")
        st.markdown(f"**Topic**: {brief.topic}")
        st.markdown(f"**Scope**: {brief.scope}")
        st.markdown(f"**Constraints**: {', '.join(brief.constraints) if brief.constraints else 'None'}")
        st.markdown(f"**Target Source Count**: {brief.target_source_count}")
        
        st.write("##### Sub-questions to Research:")
        for q in brief.sub_questions:
            st.markdown(f"- `{q.id}`: {q.question}")
        st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.subheader("⚙️ Settings")
    target_count = st.slider("Target Source Count", min_value=5, max_value=50, value=20)
    
    st.markdown("---")
    
    st.subheader("📊 Session Statistics")
    
    total_calls = st.session_state.cost_stats['vertex_calls'] + st.session_state.cost_stats['freellm_calls']
    st.metric(label="Total API Calls", value=f"{total_calls}")
    
    st.markdown("**Vertex AI (Gemini 3.5 Flash)**")
    st.write(f"Calls: {st.session_state.cost_stats['vertex_calls']}")
    st.write(f"Tokens: {st.session_state.cost_stats['vertex_input']} In / {st.session_state.cost_stats['vertex_output']} Out")
    st.write(f"Cost: **${st.session_state.cost_stats['vertex_cost']:.5f}**")
    
    st.markdown("**FreeLLMAPI**")
    st.write(f"Calls: {st.session_state.cost_stats['freellm_calls']}")
    st.write(f"Tokens: {st.session_state.cost_stats['freellm_input']} In / {st.session_state.cost_stats['freellm_output']} Out")
    st.write(f"Cost: **$0.00000**")
