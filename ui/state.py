import streamlit as st
from core.models import GraphState

def init_state() -> None:
    """Initialize session state variables for the Streamlit UI."""
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "clarification_response" not in st.session_state:
        st.session_state.clarification_response = ""
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = GraphState()
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "cost_stats" not in st.session_state:
        st.session_state.cost_stats = {
            "vertex_calls": 0,
            "vertex_input": 0,
            "vertex_output": 0,
            "vertex_cost": 0.0,
            "freellm_calls": 0,
            "freellm_input": 0,
            "freellm_output": 0,
            "freellm_cost": 0.0,
        }
