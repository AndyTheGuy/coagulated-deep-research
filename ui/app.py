import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any
import streamlit as st
import time
import asyncio
import textwrap
from ui.state import init_state
from core.graph import compile_graph
from core.models import ResearchBrief, GraphState, DREMEvaluation
from ui.log_streamer import read_logs_from_file
from core.nodes.scoping import get_router
from ui.utils import (
    update_graph_state_with_chunk as utils_update_chunk,
    update_cost_stats_from_state as utils_update_stats,
    estimate_remaining_cost as utils_estimate_remaining
)

# 1. Page config
st.set_page_config(
    page_title="Ultimate Deep Researcher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
init_state()

try:
    from config.settings import set_mock_llm_enabled
except ImportError:
    import importlib
    settings_module = importlib.import_module("config.settings")
    importlib.reload(settings_module)
    from config.settings import set_mock_llm_enabled

# Resolve Mock LLM environment variable and thread-local state from session state *before* running any graph blocking loop
if "mock_llm_mode" in st.session_state:
    is_mock = st.session_state["mock_llm_mode"]
    set_mock_llm_enabled(is_mock)
    os.environ["MOCK_LLM"] = "true" if is_mock else "false"
else:
    # Sync environment variable to session state on initial load
    is_enabled = os.environ.get("MOCK_LLM") == "true"
    st.session_state["mock_llm_mode"] = is_enabled
    set_mock_llm_enabled(is_enabled)

# 2. Custom CSS for Premium Glassmorphic UI Aesthetics
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: radial-gradient(circle at 50% 50%, #0d1117 0%, #07090e 100%) !important;
        color: #e6edf3 !important;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #f0f6fc !important;
    }
    
    .main-title {
        background: linear-gradient(135deg, #7f56da 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
        text-align: center;
    }
    
    .subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Premium Glassmorphic Card styling */
    .glass-card {
        background: rgba(22, 28, 38, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(127, 86, 218, 0.18);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        transition: border-color 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(127, 86, 218, 0.35);
    }
    
    /* Scrollbox of colorized agent logs */
    .log-box {
        background-color: #080a0f;
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 10px;
        padding: 15px;
        height: 380px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        line-height: 1.6;
        color: #c9d1d9;
        white-space: pre-wrap;
        box-shadow: inset 0 4px 12px rgba(0, 0, 0, 0.7);
    }
    
    /* Stats Cards and Tickers */
    .stat-ticker {
        text-align: center;
        padding: 14px;
        background: rgba(10, 12, 18, 0.7);
        border: 1px solid rgba(127, 86, 218, 0.15);
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
        font-weight: 500;
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #58a6ff;
    }
    
    .stat-value.neon {
        color: #00f2fe;
        text-shadow: 0 0 8px rgba(0, 242, 254, 0.3);
    }
    
    /* Status Badge Styling */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 6px;
    }
    
    .badge-success {
        background: rgba(46, 160, 67, 0.15) !important;
        color: #3fb950 !important;
        border: 1px solid rgba(46, 160, 67, 0.3) !important;
    }
    
    .badge-warning {
        background: rgba(210, 153, 34, 0.15) !important;
        color: #d29922 !important;
        border: 1px solid rgba(210, 153, 34, 0.3) !important;
    }
    
    .badge-danger {
        background: rgba(248, 81, 73, 0.15) !important;
        color: #f85149 !important;
        border: 1px solid rgba(248, 81, 73, 0.3) !important;
    }
    
    .badge-info {
        background: rgba(56, 139, 253, 0.15) !important;
        color: #58a6ff !important;
        border: 1px solid rgba(56, 139, 253, 0.3) !important;
    }
    
    /* Tabs custom appearance */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(22, 28, 38, 0.4);
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        color: #8b949e;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f0f6fc;
        border-color: rgba(127, 86, 218, 0.4);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(127, 86, 218, 0.15);
        border-color: rgba(127, 86, 218, 0.5);
        color: #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

# Main Title and Subtitle
st.markdown('<div class="main-title">🧬 Ultimate Deep Researcher</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Autonomous, self-correcting deep research agent framework powered by LangGraph</div>', unsafe_allow_html=True)

# 3. Helper Functions

def update_graph_state_with_chunk(chunk: dict) -> None:
    """Merge streamed node state dictionaries into session_state.graph_state."""
    utils_update_chunk(st.session_state.graph_state, chunk)

def update_cost_stats_from_state() -> None:
    """Pull token usage statistics from graph state and compute financial costs."""
    utils_update_stats(st.session_state.graph_state, st.session_state.cost_stats)

def estimate_remaining_cost() -> float:
    """Compute an estimated remaining cost based on pending/in_progress sub-questions."""
    return utils_estimate_remaining(st.session_state.graph_state, st.session_state.cost_stats)

def run_async_safely(coro) -> Any:
    """Safely execute an async coroutine on a fresh, dedicated event loop to prevent event loop / threading clashes."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.close()
        except Exception:
            pass

# Side panel stats renderer
def render_sidebar_html() -> str:
    cost_stats = st.session_state.cost_stats
    total_calls = cost_stats["vertex_calls"] + cost_stats["freellm_calls"]
    total_tokens = cost_stats["vertex_input"] + cost_stats["vertex_output"] + cost_stats["freellm_input"] + cost_stats["freellm_output"]
    est_rem = estimate_remaining_cost()
    
    html = f"""
    <div class="glass-card">
        <h5 style="text-align: center; color: #7f56da; margin-bottom: 15px; border-bottom: 1px solid rgba(127, 86, 218, 0.2); padding-bottom: 8px;">LLM Router Cost Ticker</h5>
        
        <div class="stat-ticker">
            <div class="stat-label">Cumulative Cost</div>
            <div class="stat-value neon">${cost_stats['vertex_cost']:.5f}</div>
        </div>
        
        <div class="stat-ticker">
            <div class="stat-label">Estimated Remaining</div>
            <div class="stat-value" style="color: #ffaa66;">${est_rem:.5f}</div>
        </div>
        
        <div style="margin-top: 15px; font-size: 0.85rem;">
            <p><b>Total API Invocations</b>: <code>{total_calls}</code></p>
            <p><b>Total Token Volume</b>: <code>{total_tokens:,}</code></p>
            <p><b>FreeLLM Failovers</b>: <code style="color: #ff9933;">{cost_stats.get('failovers', 0)}</code></p>
        </div>
        
        <hr style="border-top: 1px solid rgba(255,255,255,0.05); margin: 15px 0;">
        
        <h6 style="color: #58a6ff; font-size: 0.9rem; margin-bottom: 8px;">Vertex AI (Gemini 3.5 Flash)</h6>
        <div style="font-size: 0.8rem; line-height: 1.5; color: #8b949e;">
            Calls: <code>{cost_stats['vertex_calls']}</code><br>
            Input: <code>{cost_stats['vertex_input']:,}</code> tok<br>
            Output: <code>{cost_stats['vertex_output']:,}</code> tok
        </div>
        
        <hr style="border-top: 1px solid rgba(255,255,255,0.05); margin: 15px 0;">
        
        <h6 style="color: #66ff66; font-size: 0.9rem; margin-bottom: 8px;">FreeLLMAPI (Failover-ready)</h6>
        <div style="font-size: 0.8rem; line-height: 1.5; color: #8b949e;">
            Calls: <code>{cost_stats['freellm_calls']}</code><br>
            Input: <code>{cost_stats['freellm_input']:,}</code> tok<br>
            Output: <code>{cost_stats['freellm_output']:,}</code> tok<br>
            Cost: <code>$0.00000</code>
        </div>
    </div>
    """
    cleaned_html = "\n".join(line.strip() for line in html.splitlines())
    return cleaned_html


async def run_deep_research(query_str: str, target_count: int, constraints_str: str, status_placeholder):
    # Parse constraints
    constraints_list = [c.strip() for c in constraints_str.split(",") if c.strip()]
    
    # Setup initial state
    initial_state = {
        "user_query": query_str,
        "topic": query_str[:50],
        "sub_questions_state": [],
        "search_results": [],
        "verified_sources": [],
        "claims": [],
        "errors": [],
        "logs": [],
        "token_usage": {
            "vertex_ai": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "freellmapi": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "failovers": 0
        }
    }
    
    # Reset in-memory / state trace
    st.session_state.graph_state = GraphState()
    st.session_state.user_query = query_str
    
    # Clear previous log file for a fresh trace
    if os.path.exists("agent.json.log"):
        try:
            with open("agent.json.log", "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass
    
    app = compile_graph()
    
    with status_placeholder.container():
        live_status = st.status("🧬 Executing Autonomous Research Graph...", expanded=True)
        with live_status:
            queue_placeholder = st.empty()
            try:
                async for chunk in app.astream(initial_state, stream_mode="updates"):
                    # 1. Update main GraphState inside session state
                    update_graph_state_with_chunk(chunk)
                    
                    # 2. Extract and format logs to show live agent Trace
                    html_logs = read_logs_from_file("agent.json.log")
                    if html_logs:
                        log_content = "\n".join(html_logs)
                        log_live_placeholder.markdown(f'<div class="log-box">{log_content}</div>', unsafe_allow_html=True)
                        
                    # 3. Synchronize and display cost metrics live in sidebar
                    update_cost_stats_from_state()
                    stats_placeholder.markdown(render_sidebar_html(), unsafe_allow_html=True)
                    
                    # 4. Display current state of the parallel research queue
                    sub_qs = getattr(st.session_state.graph_state, "sub_questions_state", []) or []
                    if sub_qs:
                        queue_html = "##### 📋 Sub-questions Status Queue:\n"
                        for q in sub_qs:
                            badge_type = "badge-warning" if q.status == "pending" else "badge-info" if q.status == "in_progress" else "badge-success" if q.status == "completed" else "badge-danger"
                            queue_html += f"- **{q.id}**: {q.question} <span class='badge {badge_type}'>{q.status}</span>\n"
                        queue_placeholder.markdown(queue_html, unsafe_allow_html=True)
                            
                live_status.update(label="✅ Workflow Execution Finished Successfully!", state="complete")
                
            except Exception as e:
                live_status.update(label="❌ Workflow Execution Failed", state="error")
                st.error(f"Error during graph execution: {str(e)}")
                st.session_state.graph_state.errors.append(str(e))
    
    # Final page rerun to render tabs in completed state
    st.rerun()


async def resume_with_clarification(status_placeholder):
    question = st.session_state.graph_state.clarification_question
    clarification_response = st.session_state.clarification_response
    
    initial_state = {
        "user_query": st.session_state.user_query,
        "clarification_question": question,
        "clarification_response": clarification_response,
        "clarification_needed": False,
        "sub_questions_state": [],
        "search_results": [],
        "verified_sources": [],
        "claims": [],
        "errors": [],
        "logs": [],
        "token_usage": st.session_state.graph_state.token_usage
    }
    
    app = compile_graph()
    
    with status_placeholder.container():
        live_status = st.status("🧬 Resuming Deep Research Workflow...", expanded=True)
        with live_status:
            queue_placeholder = st.empty()
            try:
                async for chunk in app.astream(initial_state, stream_mode="updates"):
                    update_graph_state_with_chunk(chunk)
                    
                    html_logs = read_logs_from_file("agent.json.log")
                    if html_logs:
                        log_content = "\n".join(html_logs)
                        log_live_placeholder.markdown(f'<div class="log-box">{log_content}</div>', unsafe_allow_html=True)
                        
                    update_cost_stats_from_state()
                    stats_placeholder.markdown(render_sidebar_html(), unsafe_allow_html=True)
                    
                    sub_qs = getattr(st.session_state.graph_state, "sub_questions_state", []) or []
                    if sub_qs:
                        queue_html = "##### 📋 Sub-questions Status Queue:\n"
                        for q in sub_qs:
                            badge_type = "badge-warning" if q.status == "pending" else "badge-info" if q.status == "in_progress" else "badge-success" if q.status == "completed" else "badge-danger"
                            queue_html += f"- **{q.id}**: {q.question} <span class='badge {badge_type}'>{q.status}</span>\n"
                        queue_placeholder.markdown(queue_html, unsafe_allow_html=True)
                            
                live_status.update(label="✅ Workflow Execution Finished Successfully!", state="complete")
                
            except Exception as e:
                live_status.update(label="❌ Workflow Execution Failed", state="error")
                st.error(f"Error during graph execution: {str(e)}")
                st.session_state.graph_state.errors.append(str(e))
        
    st.rerun()


# 4. Layout: Left Main Column and Right Statistics Column
col_main, col_stats = st.columns([3, 1])

with col_main:
    # Set up tab containers
    tab_scope, tab_dials, tab_report = st.tabs(["🔍 Research Scope", "🎯 DREAM Quality Dials", "📄 Final Report"])
    
    with tab_scope:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔍 Research Target Setup")
        
        user_query = st.text_area(
            "Topic / Hypothesis of Interest",
            value=st.session_state.user_query,
            placeholder="Type a research query, e.g. Impact of Room-temperature Superconductors on Power Grids..."
        )
        
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            constraints_input = st.text_input(
                "Custom Scoping Constraints (comma-separated)",
                placeholder="e.g. exclude patents, prioritize peer-reviewed journals 2024+"
            )
        with col_c2:
            target_source_count = st.slider("Target Source Count", min_value=5, max_value=50, value=20)
            
        start_btn = st.button("Start Deep Research", type="primary")
        status_container_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        # Handle button click trigger (deferred to bottom once all placeholders are rendered)
        if start_btn and user_query:
            st.session_state.run_triggered = True

        # Inline Clarification dialog
        state = st.session_state.graph_state
        if state and state.clarification_needed:
            question = state.clarification_question
            st.markdown('<div class="glass-card" style="border-color: rgba(210, 153, 34, 0.4);">', unsafe_allow_html=True)
            st.warning(f"🤔 Scoping Agent requested clarification:\n\n**{question}**")
            
            clarification_response = st.text_input("Your Response:", value=st.session_state.clarification_response)
            submit_clarification = st.button("Submit Clarification", type="primary")
            clarification_status_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submit_clarification and clarification_response:
                st.session_state.clarification_response = clarification_response
                st.session_state.resume_triggered = True

        # Research Brief Viewer (Rendered statically when complete)
        if state and state.research_brief:
            brief: ResearchBrief = state.research_brief
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📋 Finalized Research Brief")
            st.markdown(f"**Topic**: {brief.topic}")
            st.markdown(f"**Scope Summary**: {brief.scope}")
            st.markdown(f"**Target Sources**: `{brief.target_source_count}` sources minimum")
            st.markdown(f"**Target Constraints**: `{', '.join(brief.constraints) if brief.constraints else 'None'}`")
            
            st.write("##### Refined Research Questions Queue:")
            sub_qs = getattr(state, "sub_questions_state", []) or brief.sub_questions
            for q in sub_qs:
                badge_type = "badge-warning" if q.status == "pending" else "badge-info" if q.status == "in_progress" else "badge-success" if q.status == "completed" else "badge-danger"
                st.markdown(f"- **{q.id}**: {q.question} <span class='badge {badge_type}'>{q.status}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_dials:
        # DREAM Quality Dials Tab
        state = st.session_state.graph_state
        if state and state.evaluation:
            eval_report: DREMEvaluation = state.evaluation
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎯 DREAM Quality Gate Evaluation")
            
            # Overall Status Badge
            if eval_report.overall_passed:
                st.markdown("<span class='badge badge-success' style='font-size: 1.15rem; padding: 6px 14px; margin-bottom: 20px;'>DREAM Gate: PASSED</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge badge-danger' style='font-size: 1.15rem; padding: 6px 14px; margin-bottom: 20px;'>DREAM Gate: REJECTED (Remediation Triggered)</span>", unsafe_allow_html=True)
            
            st.write("")
            col_kic, col_rq, col_fact = st.columns(3)
            
            with col_kic:
                st.markdown('<div class="stat-ticker">', unsafe_allow_html=True)
                st.markdown("<div class='stat-label'>Key-Info Coverage (KIC)</div>", unsafe_allow_html=True)
                kic = eval_report.key_information_coverage
                color_class = "neon" if kic.passed else ""
                st.markdown(f"<div class='stat-value {color_class}'>{kic.score:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 0.8rem; color: #8b949e;'>Threshold: {kic.threshold:.2f}</span><br>", unsafe_allow_html=True)
                badge_class = "badge-success" if kic.passed else "badge-danger"
                st.markdown(f"<span class='badge {badge_class}'>{'PASS' if kic.passed else 'FAIL'}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_rq:
                st.markdown('<div class="stat-ticker">', unsafe_allow_html=True)
                st.markdown("<div class='stat-label'>Reasoning Quality (RQ)</div>", unsafe_allow_html=True)
                rq = eval_report.reasoning_quality
                color_class = "neon" if rq.passed else ""
                st.markdown(f"<div class='stat-value {color_class}'>{rq.score:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 0.8rem; color: #8b949e;'>Threshold: {rq.threshold:.2f}</span><br>", unsafe_allow_html=True)
                badge_class = "badge-success" if rq.passed else "badge-danger"
                st.markdown(f"<span class='badge {badge_class}'>{'PASS' if rq.passed else 'FAIL'}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_fact:
                st.markdown('<div class="stat-ticker">', unsafe_allow_html=True)
                st.markdown("<div class='stat-label'>Citation Factuality</div>", unsafe_allow_html=True)
                fact = eval_report.factuality
                color_class = "neon" if fact.passed else ""
                st.markdown(f"<div class='stat-value {color_class}'>{fact.score:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 0.8rem; color: #8b949e;'>Threshold: {fact.threshold:.2f}</span><br>", unsafe_allow_html=True)
                badge_class = "badge-success" if fact.passed else "badge-danger"
                st.markdown(f"<span class='badge {badge_class}'>{'PASS' if fact.passed else 'FAIL'}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            if eval_report.evaluator_notes:
                st.markdown('<div class="glass-card" style="margin-top: 15px; border-color: rgba(255,255,255,0.05);">', unsafe_allow_html=True)
                st.markdown("##### 📝 Evaluator Commentary")
                st.markdown(eval_report.evaluator_notes)
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🎯 DREAM Quality Evaluation will execute automatically once the final report is compiled.")

    with tab_report:
        # Final Report Viewer
        state = st.session_state.graph_state
        if state and state.final_report:
            report = state.final_report
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(f"📄 {report.title}")
            
            # Confidence Summary Badge
            st.markdown(f"<span class='badge badge-success' style='font-size: 0.9rem; padding: 5px 12px;'>Overall Report Confidence: {report.confidence_score:.2f}</span>", unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown(report.content, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif state and state.draft_report:
            report = state.draft_report
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.warning("⚠️ Rendering draft report. The verifier node is currently evaluating and running DREAM quality checks...")
            st.subheader(f"📄 {report.title} (Draft)")
            st.markdown("---")
            st.markdown(report.content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📄 Finalized academic-grade research report will be displayed here once generated.")

    # 5. Progress Log Box (Underneath tabs, visible to track execution)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📜 Live Agent Trace Logs")
    log_live_placeholder = st.empty()
    
    # Init log box view with current files
    html_logs = read_logs_from_file("agent.json.log")
    if html_logs:
        log_content = "\n".join(html_logs)
        log_live_placeholder.markdown(f'<div class="log-box">{log_content}</div>', unsafe_allow_html=True)
    else:
        log_live_placeholder.info("Agent execution logs will appear here when research starts.")
    st.markdown('</div>', unsafe_allow_html=True)


with col_stats:
    st.markdown('<h3 style="text-align: center; margin-bottom: 20px;">📊 Statistics</h3>', unsafe_allow_html=True)
    
    st.checkbox(
        "Enable Offline Mock Mode",
        key="mock_llm_mode",
        help="Run the entire research workflow offline using high-fidelity mock LLM generation (perfect for quick demonstration, zero API costs, zero internet requirements)."
    )
    
    stats_placeholder = st.empty()
    
    # Initial render of stats sidebar
    update_cost_stats_from_state()
    stats_placeholder.markdown(render_sidebar_html(), unsafe_allow_html=True)


# 6. Execute deferred background workflows at the very bottom once all placeholders are rendered
if getattr(st.session_state, "run_triggered", False):
    st.session_state.run_triggered = False  # Reset flag
    q = user_query if 'user_query' in locals() else st.session_state.get("user_query", "")
    c = constraints_input if 'constraints_input' in locals() else ""
    t = target_source_count if 'target_source_count' in locals() else 20
    placeholder = status_container_placeholder if 'status_container_placeholder' in locals() else st.empty()
    run_async_safely(run_deep_research(q, t, c, placeholder))

if getattr(st.session_state, "resume_triggered", False):
    st.session_state.resume_triggered = False  # Reset flag
    placeholder = clarification_status_placeholder if 'clarification_status_placeholder' in locals() else st.empty()
    run_async_safely(resume_with_clarification(placeholder))
