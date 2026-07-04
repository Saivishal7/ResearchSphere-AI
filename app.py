# app.py
# ResearchSphere AI - Premium Streamlit UI Shell (Phase 1)
# Built with modern SaaS aesthetics, fluid layout, responsive components, and custom CSS

import os
import datetime
import logging
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PAGE SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ResearchSphere AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================
def initialize_session_state():
    """Initializes only missing global state variables required by Phase 1."""
    if "role" not in st.session_state:
        st.session_state.role = "Student"
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "current_results" not in st.session_state:
        st.session_state.current_results = None
    if "approval_state" not in st.session_state:
        st.session_state.approval_state = {
            "status": "Pending Verification",
            "action": "Dispatch Introduction Request",
            "reason": "Outstanding academic alignment in Explainable AI and Medical Diagnostics.",
            "affected_faculty": ["Dr. Sarah Jenkins", "Dr. Alan Turing"],
            "generated_content": (
                "Subject: Academic Partnership Proposal: Explainable Diagnostic Models\n\n"
                "Dear Dr. Jenkins and Dr. Turing,\n\n"
                "I am writing to propose a potential research collaboration under the "
                "ResearchSphere AI umbrella. \n\n"
                "We identified high similarity between your medical computer vision models "
                "and Dr. Turing's self-explaining transformer layers. Combining these fields "
                "could yield significant advancements in trusted diagnostic intelligence.\n\n"
                "Would you be open to a 15-minute sync next Tuesday at 14:00 UTC?\n\n"
                "Sincerely,\nResearchSphere Orchestrator"
            )
        }
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []

initialize_session_state()


def _get_backend_status() -> Dict[str, Any]:
    """Collects live backend availability flags for the dashboard and settings views."""
    try:
        from config import GEMINI_API_KEY, TAVILY_API_KEY, CHROMA_STORE_DIR
    except Exception:
        GEMINI_API_KEY = None
        TAVILY_API_KEY = None
        CHROMA_STORE_DIR = "chroma_store"

    faculty_dir = os.path.join("data", "faculty")
    try:
        from ingestion.load_faculty import load_all_faculty_profiles
        profiles = load_all_faculty_profiles(faculty_dir)
        faculty_dataset_exists = bool(profiles)
    except Exception:
        faculty_dataset_exists = False

    try:
        from graph.build_graph import compiled_research_graph
        langgraph_compiled = compiled_research_graph is not None
    except Exception:
        langgraph_compiled = False

    return {
        "gemini_configured": bool(GEMINI_API_KEY),
        "tavily_configured": bool(TAVILY_API_KEY),
        "chroma_configured": os.path.isdir(CHROMA_STORE_DIR) and os.listdir(CHROMA_STORE_DIR) != [] if os.path.isdir(CHROMA_STORE_DIR) else False,
        "faculty_dataset_exists": faculty_dataset_exists,
        "langgraph_compiled": langgraph_compiled,
    }


def _collect_dashboard_metrics() -> Dict[str, Any]:
    """Builds dashboard values from the local data and backend modules."""
    try:
        from ingestion.load_faculty import load_all_faculty_profiles
        profiles = load_all_faculty_profiles(os.path.join("data", "faculty"))
    except Exception:
        profiles = []

    research_areas = sorted({interest.lower() for profile in profiles for interest in profile.research_interests})
    agent_files = [name for name in os.listdir("agents") if name.endswith(".py") and name != "__init__.py"]
    backend_status = _get_backend_status()

    return {
        "faculty_count": len(profiles),
        "research_area_count": len(research_areas),
        "agent_count": len(agent_files),
        "backend_status": backend_status,
    }


def _run_langgraph_query(query: str, role: str) -> Dict[str, Any]:
    """Executes the existing LangGraph workflow for student or professor requests."""
    try:
        from graph.build_graph import compiled_research_graph
    except Exception as exc:
        raise RuntimeError(f"LangGraph could not be compiled: {exc}") from exc

    state: Dict[str, Any] = {
        "current_query": query,
        "user_role": role.lower() if role.lower() in {"student", "professor"} else "unknown",
        "intent": "general_query",
        "conversation_history": [],
        "student_profile": None,
        "professor_profile": None,
        "faculty_results": None,
        "selected_faculty": None,
        "research_trends": None,
        "research_gaps": None,
        "collaboration_suggestions": None,
        "project_recommendations": None,
        "pending_action": None,
        "approval_required": False,
        "approval_status": "pending",
        "retrieved_context": None,
        "tool_output": None,
        "error": None,
        "retry_count": 0,
        "session_id": "streamlit",
    }

    return compiled_research_graph.invoke(state)


def _run_student_recommendation(query: str) -> Dict[str, Any]:
    """Runs the student backend flow and returns structured faculty matches."""
    try:
        from agents.student_agent import StudentAgent
        from agents.faculty_retrieval_agent import FacultyRetrievalAgent
    except Exception as exc:
        return {"error": f"Student backend could not be loaded: {exc}"}

    try:
        graph_result = _run_langgraph_query(query, "student")
        faculty_results = graph_result.get("faculty_results") or []
    except Exception as exc:
        faculty_results = []
        graph_error = str(exc)
    else:
        graph_error = None

    if not faculty_results:
        try:
            retrieval_result = FacultyRetrievalAgent().retrieve_relevant_faculty(query=query, k=3)
            if retrieval_result.success and retrieval_result.matches:
                faculty_results = [match.model_dump() if hasattr(match, "model_dump") else match.dict() for match in retrieval_result.matches]
        except Exception as exc:
            faculty_results = []
            graph_error = graph_error or str(exc)

    try:
        student_response = StudentAgent().search_supervisors(query)
    except Exception as exc:
        student_response = None
        student_error = str(exc)
    else:
        student_error = None

    return {
        "matches": faculty_results,
        "recommendations": [
            {"name": item.get("name", ""), "match_explanation": item.get("match_explanation", ""), "rank": item.get("rank", 1)}
            for item in (student_response.recommended_faculty if student_response else [])
        ],
        "reasoning": student_response.reasoning if student_response else "",
        "error": graph_error or student_error,
    }


def _run_professor_action(topic: str, action: str) -> Dict[str, Any]:
    """Runs the professor backend action requested by the user."""
    try:
        if action == "trend":
            from agents.trend_agent import TrendAgent
            response = TrendAgent().analyze_trends(topic)
            payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
            if payload.get("status") in {"failed", "no_results", "partial"}:
                logger.warning(f"Professor trend action returned status '{payload.get('status')}' for topic '{topic}'")
            return {
                "title": "Research Trends",
                "payload": payload,
            }
        if action == "gap":
            from agents.gap_agent import GapAnalysisAgent
            response = GapAnalysisAgent().analyze_gaps(topic)
            payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
            if payload.get("status") in {"failed", "partial"}:
                logger.warning(f"Professor gap action returned status '{payload.get('status')}' for topic '{topic}'")
            return {
                "title": "Research Gaps",
                "payload": payload,
            }
        if action == "collaboration":
            from agents.collaboration_agent import CollaborationAgent
            response = CollaborationAgent().suggest_collaborations(topic)
            return {
                "title": "Collaboration Opportunities",
                "payload": response.model_dump() if hasattr(response, "model_dump") else response.dict(),
            }
    except Exception as exc:
        return {"title": "Backend Error", "payload": {"error": str(exc)}}

    return {"title": "No action selected", "payload": {}}


def _run_project_recommendation(query: str) -> Dict[str, Any]:
    """Runs the project recommendation agent and returns a structured recommendation."""
    try:
        from agents.project_recommendation_agent import ProjectRecommendationAgent
        response = ProjectRecommendationAgent().recommend_projects(query)
        return response
    except Exception as exc:
        return {"status": "failed", "reasoning": f"Project recommendation backend failed: {exc}"}


def _build_confirmation_view() -> Dict[str, Any]:
    """Builds a live confirmation object for the approval page from the existing session state."""
    try:
        from agents.confirmation_agent import ConfirmationAgent
    except Exception as exc:
        return {"error": str(exc)}

    pending_action = {
        "action": st.session_state.approval_state.get("action", "Dispatch Introduction Request"),
        "reason": st.session_state.approval_state.get("reason", "No justification provided."),
        "affected_faculty": st.session_state.approval_state.get("affected_faculty", []),
        "generated_content": st.session_state.approval_state.get("generated_content", ""),
    }
    confirmation_agent = ConfirmationAgent()
    confirmation_obj = confirmation_agent.build_confirmation_object(pending_action)
    return {
        "action": confirmation_obj.action,
        "reason": confirmation_obj.reason,
        "affected_faculty": confirmation_obj.affected_faculty,
        "generated_content": confirmation_obj.generated_content,
    }


# ==============================================================================
# 3. PREMIUM THEME CSS INJECTION
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #2563EB;
    --secondary: #14B8A6;
    --accent: #6366F1;
    --bg: #F8FAFC;
    --surface: #FFFFFF;
    --surface-soft: #F8FBFF;
    --text: #0F172A;
    --muted: #64748B;
    --border: #E2E8F0;
    --shadow: 0 18px 45px -24px rgba(15, 23, 42, 0.35);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--text);
}

.stApp {
    background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1360px !important;
}

.custom-card {
    background: linear-gradient(145deg, var(--surface) 0%, var(--surface-soft) 100%);
    border: 1px solid rgba(226, 232, 240, 0.92);
    border-radius: 16px;
    padding: 1.25rem 1.3rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.custom-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 45px -24px rgba(37, 99, 235, 0.35);
    border-color: rgba(37, 99, 235, 0.16);
}

.metric-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid rgba(226, 232, 240, 0.95);
    border-radius: 16px;
    padding: 1.1rem 1.15rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
    min-height: 132px;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 45px -24px rgba(15, 23, 42, 0.28);
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.65rem;
}

.metric-icon {
    font-size: 1.15rem;
    color: var(--primary);
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(99, 102, 241, 0.16));
    padding: 0.45rem;
    border-radius: 10px;
    line-height: 1;
}

.metric-label {
    font-size: 0.77rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--text);
    margin: 0.2rem 0 0.2rem;
    letter-spacing: -0.03em;
}

.metric-sub {
    font-size: 0.8rem;
    color: var(--secondary);
    font-weight: 600;
}

.status-badge {
    padding: 0.32rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    background: rgba(248, 250, 252, 0.9);
    color: var(--muted);
    border: 1px solid rgba(226, 232, 240, 0.95);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.status-badge.live {
    background: rgba(20, 184, 166, 0.12);
    border-color: rgba(20, 184, 166, 0.18);
    color: #0f766e;
}

.status-badge-dot {
    width: 6px;
    height: 6px;
    background-color: var(--secondary);
    border-radius: 50%;
}

.hero-shell {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(99, 102, 241, 0.16));
    border: 1px solid rgba(37, 99, 235, 0.12);
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}

.hero-panel {
    background: linear-gradient(120deg, rgba(15, 23, 42, 0.97) 0%, rgba(30, 41, 59, 0.94) 100%);
    border-radius: 18px;
    color: #f8fafc;
    padding: 1.35rem 1.4rem;
    display: flex;
    justify-content: space-between;
    gap: 1.2rem;
    align-items: center;
    flex-wrap: wrap;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 0.6rem;
}

.hero-copy h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.hero-copy p {
    margin: 0.35rem 0 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    max-width: 700px;
}

.hero-badges {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 0.85rem;
}

.hero-meta {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
}

.meta-card {
    min-width: 160px;
    background: rgba(248, 250, 252, 0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
}

.meta-label {
    display: block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}

.meta-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
}

[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 0.7rem 0.8rem;
    margin-bottom: 0.45rem;
    text-align: left;
    background: rgba(255, 255, 255, 0.05);
    color: #f8fafc;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    transform: translateX(2px);
    background: rgba(37, 99, 235, 0.18);
    border-color: rgba(37, 99, 235, 0.35);
}

[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.24), rgba(99, 102, 241, 0.2));
    border-color: rgba(191, 219, 254, 0.18);
}

div.stButton > button {
    background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
    color: var(--text);
    border: 1px solid rgba(226, 232, 240, 0.95);
    border-radius: 999px;
    padding: 0.7rem 1rem;
    font-size: 0.92rem;
    font-weight: 600;
    transition: all 0.2s ease;
    width: 100%;
    box-shadow: 0 8px 20px -16px rgba(15, 23, 42, 0.28);
}

div.stButton > button:hover {
    border-color: rgba(37, 99, 235, 0.26);
    color: var(--primary);
    transform: translateY(-1px);
    box-shadow: 0 12px 24px -16px rgba(37, 99, 235, 0.36);
}

div.stButton > button:active {
    transform: translateY(0);
}

.hipl-alert {
    background: linear-gradient(135deg, #fff7ed 0%, #fffbeb 100%);
    border-left: 4px solid #f59e0b;
    padding: 1.15rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    box-shadow: 0 12px 28px -20px rgba(245, 158, 11, 0.28);
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. LEFT SIDEBAR (LOGO, ROLE SELECTOR, & NAVIGATION)
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.25rem; margin-bottom: 0.7rem; padding: 0.2rem 0 0.2rem;">
        <div style="width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #2563EB 0%, #6366F1 100%); box-shadow: 0 12px 25px -16px rgba(37, 99, 235, 0.55); font-size: 1.2rem;">🪐</div>
        <div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">ResearchSphere</div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.1rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700;">Multi-Agent Orchestration</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 0.65rem;'>Workspace Role</div>", unsafe_allow_html=True)

    roles_list = ["Student", "Professor", "Admin"]
    selected_role = st.selectbox(
        "User Persona Role",
        roles_list,
        index=roles_list.index(st.session_state.role) if st.session_state.role in roles_list else 0,
        label_visibility="collapsed",
        key="sidebar_role_select"
    )
    if selected_role != st.session_state.role:
        st.session_state.role = selected_role
        st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 0.6rem;'>Navigation</div>", unsafe_allow_html=True)

    pages_list = [
        "🏠 Dashboard",
        "🎓 Student Portal",
        "👨‍🏫 Professor Portal",
        "📊 Research Analytics",
        "🤝 Collaboration Center",
        "📁 Project Recommendations",
        "✅ Approval Center",
        "⚙️ Settings",
        "ℹ️ About"
    ]

    for page in pages_list:
        is_active = page == st.session_state.current_page
        if st.button(page, key=f"nav_{page}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_page = page
            st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0.8rem 0.85rem; border-radius: 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); font-size: 0.74rem; color: #94A3B8; line-height: 1.45;">
        <div><b style="color: #F8FAFC;">Node ID:</b> main-orchestrator</div>
        <div><b style="color: #F8FAFC;">Session State:</b> Local Storage</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. TOP HEADER
# ==============================================================================
current_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

st.markdown(f"""
<div class="hero-shell">
    <div class="hero-panel">
        <div class="hero-copy">
            <div class="hero-pill">ResearchSphere AI • Phase 1</div>
            <h1>ResearchSphere AI</h1>
            <p>AI-powered research collaboration workflows for students, professors, and administrators with a calmer, premium experience.</p>
            <div class="hero-badges">
                <span class="status-badge live"><span class="status-badge-dot"></span> Live shell</span>
                <span class="status-badge"><span class="status-badge-dot"></span> Multi-agent ready</span>
                <span class="status-badge"><span class="status-badge-dot"></span> Human review enabled</span>
            </div>
        </div>
        <div class="hero-meta">
            <div class="meta-card">
                <span class="meta-label">Current UTC time</span>
                <span class="meta-value">{current_time_str}</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">Active role</span>
                <span class="meta-value">{st.session_state.role}</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">Backend status</span>
                <span class="meta-value">{'Live' if _get_backend_status()['langgraph_compiled'] else 'Needs config'}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 6. ROUTER & PAGE DISPATCHER
# ==============================================================================

# ------------------------------------------------------------------------------
# PAGE A: 🏠 Dashboard
# ------------------------------------------------------------------------------
if st.session_state.current_page == "🏠 Dashboard":
    dashboard_metrics = _collect_dashboard_metrics()
    backend_status = dashboard_metrics["backend_status"]

    # 1. Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">Faculty Profiles</span>
                <span class="metric-icon">📁</span>
            </div>
            <div class="metric-value">{dashboard_metrics['faculty_count']}</div>
            <span class="metric-sub">✓ Loaded from data/faculty</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">Research Areas</span>
                <span class="metric-icon">🏷️</span>
            </div>
            <div class="metric-value">{dashboard_metrics['research_area_count']}</div>
            <span class="metric-sub">✓ Derived from faculty profiles</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">AI Agents</span>
                <span class="metric-icon">🤖</span>
            </div>
            <div class="metric-value">{dashboard_metrics['agent_count']}</div>
            <span class="metric-sub">✓ Active backend modules</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">System Status</span>
                <span class="metric-icon">⚡</span>
            </div>
            <div class="metric-value" style="color:{'#10B981' if backend_status['langgraph_compiled'] else '#64748B'};">{'Online' if backend_status['langgraph_compiled'] else 'Standby'}</div>
            <span class="metric-sub" style="color:{'#10B981' if backend_status['langgraph_compiled'] else '#64748B'};">{'Live workflow available' if backend_status['langgraph_compiled'] else 'Backend needs configuration'}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 2. Main Content Split: Quick Actions / Activity Feed vs System Monitor Side
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Quick Actions Container
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 0.25rem 0;">⚡ Quick Research Actions</h3>
            <p style="color: #64748B; font-size: 0.85rem; margin: 0 0 1rem 0;">One-click routing across multi-agent portals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔍 Find Faculty (Student)", help="Discover academic supervisors matching research criteria."):
                st.session_state.current_page = "🎓 Student Portal"
                st.rerun()
            if st.button("💡 Generate Project Recommendations", help="Synthesize thesis proposals using agent chains."):
                st.session_state.current_page = "📁 Project Recommendations"
                st.rerun()
            if st.button("📈 Explore Research Trends", help="Scrape, search, and analyze web and publication trends."):
                st.session_state.current_page = "📊 Research Analytics"
                st.rerun()
        with btn_col2:
            if st.button("🎯 Analyze Research Gaps (Professor)", help="Evaluate published contexts for novel research niches."):
                st.session_state.current_page = "👨‍🏫 Professor Portal"
                st.rerun()
            if st.button("🤝 Discover Peer Collaborations", help="Analyze and pair faculty members with complementary profiles."):
                st.session_state.current_page = "🤝 Collaboration Center"
                st.rerun()
            if st.button("✅ Access Approval Center Queue", help="Review and authorize human-in-the-loop dispatches."):
                st.session_state.current_page = "✅ Approval Center"
                st.rerun()

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Recent Activity Panel
        st.markdown("""
        <div>
            <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 1rem 0;">📝 Recent Activity</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 12px; padding: 3rem 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">📋</div>
            <h4 style="color: #475569; margin: 0 0 0.375rem 0; font-weight: 600; font-size: 1rem;">No activity yet.</h4>
            <p style="color: #64748B; font-size: 0.85rem; margin: 0; max-width: 420px; margin-left: auto; margin-right: auto;">
                Your search queries, multi-agent log streams, and approval histories will render in this real-time timeline once integrated.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        # System status panel
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="color: #0F172A; font-size: 1.25rem; font-weight: 700; margin: 0 0 0.25rem 0;">📡 Integration Core</h3>
            <p style="color: #64748B; font-size: 0.85rem; margin: 0 0 1rem 0;">Connector pipeline status check.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="custom-card" style="padding: 1.25rem;">
            <div style="display: flex; flex-direction: column; gap: 0.85rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">Gemini Service</span>
                    <span class="status-badge{' live' if backend_status['gemini_configured'] else ''}"><span class="status-badge-dot"></span>{'Live' if backend_status['gemini_configured'] else 'Offline'}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">ChromaDB Database</span>
                    <span class="status-badge{' live' if backend_status['chroma_configured'] else ''}"><span class="status-badge-dot"></span>{'Live' if backend_status['chroma_configured'] else 'Offline'}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">LangGraph Orchestrator</span>
                    <span class="status-badge{' live' if backend_status['langgraph_compiled'] else ''}"><span class="status-badge-dot"></span>{'Live' if backend_status['langgraph_compiled'] else 'Offline'}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">Tavily Search Engine</span>
                    <span class="status-badge{' live' if backend_status['tavily_configured'] else ''}"><span class="status-badge-dot"></span>{'Live' if backend_status['tavily_configured'] else 'Offline'}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">Faculty Dataset</span>
                    <span class="status-badge{' live' if backend_status['faculty_dataset_exists'] else ''}"><span class="status-badge-dot"></span>{'Live' if backend_status['faculty_dataset_exists'] else 'Offline'}</span>
                </div>
            </div>
            <div style="margin-top: 1.25rem; background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 8px; font-size: 0.75rem; color: #64748B; line-height: 1.4;">
                💡 <b>Developer Note:</b> The shell now reads live backend availability and data files from the existing project modules.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE B: 🎓 Student Portal
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "🎓 Student Portal":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">🎓 Student Advising matching & Recommendations</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Describe your research interests to match with university supervisors and generate custom project proposals.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("student_query_form"):
        student_interest = st.text_area(
            "Enter your detailed academic/research interests:",
            placeholder="e.g., Deep learning models for medical image segmentation, specifically looking at MRI datasets with explainable self-attention maps.",
            height=120,
            value=st.session_state.current_query
        )

        col_f1, col_f2 = st.columns([1, 4])
        with col_f1:
            submit_btn = st.form_submit_button("🔍 Find Matches")

        if submit_btn:
            st.session_state.current_query = student_interest
            try:
                st.session_state.current_results = _run_student_recommendation(student_interest)
            except Exception as exc:
                st.session_state.current_results = {"error": str(exc)}

    if st.session_state.current_query:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🗺️ Orchestrated Matchmaker Output")

        results = st.session_state.get("current_results") or {}
        if results.get("error"):
            st.error(f"The student backend could not complete the request: {results['error']}")
        elif results.get("matches"):
            for index, match in enumerate(results["matches"][:3], start=1):
                recommendation = next((item for item in results.get("recommendations", []) if item.get("name", "").lower() == str(match.get("name", "")).lower()), {})
                interests = ", ".join(match.get("research_interests", [])[:4]) or "No interests listed"
                publications = match.get("publications", []) or []
                publication_text = ", ".join([pub.get("title", "") for pub in publications[:2] if pub.get("title")]) or "No publications listed"
                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                        <div>
                            <span style="font-weight: 700; font-size: 1.1rem; color: #0F172A;">{match.get('name', 'Faculty')}</span><br>
                            <span style="font-size: 0.8rem; color: #64748B;">{match.get('department', 'Department')}</span>
                        </div>
                        <span style="background-color: #E0F2FE; color: #0369A1; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.5rem; border-radius: 4px;">{round(float(match.get('similarity_score', 0)) * 100, 1)}% Match Score</span>
                    </div>
                    <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 0.75rem;">
                        <b>Research Interests:</b> {interests}
                    </p>
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; color: #475569; margin-bottom: 0.75rem;">
                        <b>Matching Publications:</b> {publication_text}
                    </div>
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; color: #475569;">
                        <b>Reason for Recommendation:</b> {recommendation.get('match_explanation', results.get('reasoning', 'Recommendation generated from the backend.'))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Submit a research topic to run the student matching workflow.")

# ------------------------------------------------------------------------------
# PAGE C: 👨‍🏫 Professor Portal
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "👨‍🏫 Professor Portal":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">👨‍🏫 Professor Analytics & Strategy Hub</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Extract global academic publication trends, evaluate critical research gaps, and build co-authorship pipelines.</p>
    </div>
    """, unsafe_allow_html=True)

    prof_interest = st.text_input(
        "Enter academic discipline or topic of interest:",
        placeholder="e.g., Quantum Machine Learning or Explainable Neural Networks",
        value=st.session_state.get("professor_query", "Explainable Neural Networks")
    )

    col_prof1, col_prof2, col_prof3 = st.columns(3)
    with col_prof1:
        trend_trigger = st.button("📈 Extract Research Trends")
    with col_prof2:
        gap_trigger = st.button("🎯 Identify Research Gaps")
    with col_prof3:
        collab_trigger = st.button("🤝 Discover Potential Co-Authors")

    if trend_trigger or gap_trigger or collab_trigger:
        st.session_state.professor_query = prof_interest
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📋 Backend Output")

        if trend_trigger:
            result = _run_professor_action(prof_interest, "trend")
            payload = result.get("payload", {})
            if payload.get("error"):
                st.error(payload["error"])
            else:
                st.markdown(f"""
                <div class="custom-card">
                    <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Research Trends</h4>
                    <p style="font-size: 0.9rem; color: #475569; line-height: 1.5; margin-bottom: 0.75rem;">{payload.get('summary', 'No summary available')}</p>
                    <ul style="font-size: 0.85rem; color: #475569; line-height: 1.6; margin-bottom: 0;">
                        {''.join(f'<li><b>{item}</b></li>' for item in payload.get('emerging_topics', [])[:4])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        elif gap_trigger:
            result = _run_professor_action(prof_interest, "gap")
            payload = result.get("payload", {})
            if payload.get("error"):
                st.error(payload["error"])
            else:
                st.markdown(f"""
                <div class="custom-card" style="border-left: 4px solid #EF4444;">
                    <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Research Gaps</h4>
                    <ul style="font-size: 0.85rem; color: #475569; line-height: 1.6; margin-bottom: 0;">
                        {''.join(f'<li><b>{item}</b></li>' for item in payload.get('identified_gaps', [])[:4])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        elif collab_trigger:
            result = _run_professor_action(prof_interest, "collaboration")
            payload = result.get("payload", {})
            if payload.get("error"):
                st.error(payload["error"])
            else:
                collaborators = payload.get("recommended_collaborators", [])
                if collaborators:
                    for collaborator in collaborators[:3]:
                        st.markdown(f"""
                        <div class="custom-card" style="border-left: 4px solid #10B981;">
                            <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">{collaborator.get('name', 'Collaborator')}</h4>
                            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 0.5rem;">
                                <b>Department:</b> {collaborator.get('department', 'Unknown')}<br>
                                <b>Specialty:</b> {collaborator.get('specialty', 'Unknown')}
                            </p>
                            <p style="font-size: 0.85rem; color: #64748B; line-height: 1.5; margin: 0;">
                                <b>Reason:</b> {collaborator.get('complementary_strength', 'No reason supplied.')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No collaboration matches were returned by the backend.")

# ------------------------------------------------------------------------------
# PAGE D: 📊 Research Analytics
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "📊 Research Analytics":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">📊 Research Analytics</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Academic publication metrics, co-citation statistics, and domain density graphs.</p>
    </div>
    """, unsafe_allow_html=True)

    # Render beautiful charts with mock data using Streamlit's native functions
    st.markdown("### 📈 Publication Trends by Domain (2018 - 2026)")
    years = [str(y) for y in range(2018, 2027)]
    
    chart_data = pd.DataFrame({
        "Computer Vision": [120, 150, 190, 240, 310, 390, 480, 580, 680],
        "Natural Language Processing": [140, 180, 250, 340, 420, 510, 630, 750, 920],
        "Explainable AI (XAI)": [20, 35, 55, 90, 140, 210, 290, 410, 560]
    }, index=years)
    
    st.line_chart(chart_data)

    col_ar1, col_ar2 = st.columns(2)
    with col_ar1:
        st.markdown("### 📊 Distribution of Citations")
        cite_data = pd.DataFrame({
            "Topic": ["Transformers", "CNNs", "LLMs", "RAG Pipeline", "XAI Bottlenecks", "GNNs"],
            "Citations (Thousands)": [45.2, 38.1, 32.8, 21.5, 14.2, 8.9]
        }).set_index("Topic")
        st.bar_chart(cite_data)
        
    with col_ar2:
        st.markdown("""
        <div class="custom-card" style="height: 100%;">
            <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Vector Density & Ingestion Log</h4>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.6;">
                <p style="margin: 0 0 0.5rem 0;"><b>Total Vector Indexes:</b> 1,540 Documents</p>
                <p style="margin: 0 0 0.5rem 0;"><b>Embedding Dimension:</b> 384 (all-MiniLM-L6-v2)</p>
                <p style="margin: 0 0 0.5rem 0;"><b>Vector Store:</b> ChromaDB (Local Sandbox Container)</p>
                <p style="margin: 0 0 0.75rem 0;"><b>Update Frequency:</b> On-demand ingestion via <code>build_vectordb.py</code></p>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 6px; font-size: 0.75rem; color: #64748B; font-family: monospace;">
                    [CHROMA] Loaded 15 JSON files from data/faculty/<br>
                    [CHROMA] Embedded 82 chunks using SentenceTransformer
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE E: 🤝 Collaboration Center
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "🤝 Collaboration Center":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">🤝 Collaboration Center</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Cross-disciplinary research pairings and visual network matches formulated by the CollaborationAgent.</p>
    </div>
    """, unsafe_allow_html=True)

    # Show a couple of premium collaboration mockup cards
    st.markdown("### Recommended Multi-Disciplinary Pairings")
    
    col_co1, col_co2 = st.columns(2)
    with col_co1:
        st.markdown("""
        <div class="custom-card" style="border-top: 4px solid #3B82F6;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #3B82F6; background-color: #EFF6FF; padding: 0.2rem 0.5rem; border-radius: 4px;">94% Complementary Score</span>
                <span style="font-size: 0.8rem; color: #64748B; font-weight: 500;">XAI + Vision Diagnostics</span>
            </div>
            <h4 style="margin: 0 0 0.5rem 0; color: #0F172A; font-weight:700;">Dr. Sarah Jenkins & Dr. Alan Turing</h4>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Jenkins' deep dataset on oncology scans represents an outstanding validation sandbox for Turing's novel self-explaining layer configurations.
            </p>
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; color: #475569;">
                <b>Project Proposal Target:</b> "Explainable Vision Transformers for Diagnostic Oncology Verification", NIH Grant Track.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_co2:
        st.markdown("""
        <div class="custom-card" style="border-top: 4px solid #10B981;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #10B981; background-color: #ECFDF5; padding: 0.2rem 0.5rem; border-radius: 4px;">88% Complementary Score</span>
                <span style="font-size: 0.8rem; color: #64748B; font-weight: 500;">QML + Privacy Physics</span>
            </div>
            <h4 style="margin: 0 0 0.5rem 0; color: #0F172A; font-weight:700;">Dr. Diana Prince & Dr. Bruce Wayne</h4>
            <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                Wayne's research into cryptographic model structures pairs natively with Prince's quantum machine learning framework, creating quantum-safe privacy barriers.
            </p>
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; color: #475569;">
                <b>Project Proposal Target:</b> "Quantum-Safe Federated Learning Architectures for Classified Distributed Nodes".
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE F: 📁 Project Recommendations
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "📁 Project Recommendations":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">📁 Project Recommendations</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Orchestrated research projects and grant-ready thesis suggestions generated via multi-agent synthesis.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("project_recommendation_form"):
        project_query = st.text_input(
            "Describe your research interests or target project area:",
            value=st.session_state.get("project_query", "Explainable AI for medical diagnostics")
        )
        submitted = st.form_submit_button("Generate Recommendation")

    if submitted:
        st.session_state.project_query = project_query
        st.session_state.project_recommendation_result = _run_project_recommendation(project_query)

    result = st.session_state.get("project_recommendation_result")
    if result:
        st.markdown("### Generated Thesis Proposal")
        if result.get("status") == "failed":
            st.error(result.get("reasoning", "The project recommendation backend failed."))
        else:
            st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <h3 style="margin:0 0 0.25rem 0; font-size:1.2rem; color:#0F172A; font-weight:700;">{result.get('project_title', 'Research Proposal')}</h3>
                        <span style="font-size: 0.8rem; color: #64748B;">{result.get('difficulty', 'Intermediate')} · {result.get('status', 'partial')}</span>
                    </div>
                    <span style="background-color: #F1F5F9; color: #475569; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.5rem; border-radius: 4px;">{result.get('required_skills', ['Research'])[:2]}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem; margin-bottom: 1rem;">
                    <div>
                        <h5 style="margin:0 0 0.375rem 0; font-weight:600; color:#475569; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.02em;">Problem Statement & Gap</h5>
                        <p style="font-size:0.85rem; color:#1E293B; line-height:1.5; margin:0;">{result.get('problem_statement', 'No problem statement returned.')}</p>
                    </div>
                    <div>
                        <h5 style="margin:0 0 0.375rem 0; font-weight:600; color:#475569; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.02em;">Methodology</h5>
                        <p style="font-size:0.85rem; color:#1E293B; line-height:1.5; margin:0;">{result.get('methodology', 'No methodology returned.')}</p>
                    </div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.85rem; border-radius: 8px; font-size: 0.8rem; color: #475569; line-height: 1.4;">
                    <b>Reasoning:</b> {result.get('reasoning', 'No reasoning returned.')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Generate a recommendation to invoke the project recommendation agent.")

# ------------------------------------------------------------------------------
# PAGE G: ✅ Approval Center
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "✅ Approval Center":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">✅ Human-In-The-Loop Approval Center</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Review, modify, and authorize high-consequence multi-agent actions (such as sending introduction emails or dispatching API calls).</p>
    </div>
    """, unsafe_allow_html=True)

    confirmation_view = _build_confirmation_view()
    if confirmation_view.get("error"):
        st.error(f"The confirmation backend could not be loaded: {confirmation_view['error']}")

    st.markdown(f"""
    <div class="hipl-alert">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <b style="color: #92400E; font-size: 1.1rem;">⚠️ Verification Required: {confirmation_view.get('action', st.session_state.approval_state['action'])}</b>
            <span style="background-color: #FEF3C7; color: #B45309; border: 1px solid #F59E0B; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">
                {st.session_state.approval_state['status'].upper()}
            </span>
        </div>
        <div style="color: #78350F; font-size: 0.85rem; line-height: 1.5;">
            <b>Justification:</b> {confirmation_view.get('reason', st.session_state.approval_state['reason'])}<br>
            <b>Target Faculty Node Pair:</b> {', '.join(confirmation_view.get('affected_faculty', st.session_state.approval_state['affected_faculty']))}
        </div>
    </div>
    """, unsafe_allow_html=True)

    edited_text = st.text_area(
        "Edit Draft Communication / Command dispatch:",
        value=confirmation_view.get("generated_content", st.session_state.approval_state["generated_content"]),
        height=240,
        disabled=(st.session_state.approval_state["status"] != "Pending Verification")
    )

    if edited_text != st.session_state.approval_state["generated_content"]:
        st.session_state.approval_state["generated_content"] = edited_text

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("👍 APPROVE & AUTHORIZE DISPATCH", disabled=(st.session_state.approval_state["status"] != "Pending Verification")):
            st.session_state.approval_state["status"] = "Approved & Sent"
            st.success("✅ Action authorized. The confirmation agent now reflects the live approval state.")
            st.rerun()
    with btn_col2:
        if st.button("❌ REJECT & DISCARD ACTION", disabled=(st.session_state.approval_state["status"] != "Pending Verification")):
            st.session_state.approval_state["status"] = "Rejected & Discarded"
            st.error("❌ Action rejected and removed from current queue.")
            st.rerun()
    with btn_col3:
        if st.button("🔄 RESET QUEUE DEMO"):
            st.session_state.approval_state["status"] = "Pending Verification"
            st.session_state.approval_state["generated_content"] = (
                "Subject: Academic Partnership Proposal: Explainable Diagnostic Models\n\n"
                "Dear Dr. Jenkins and Dr. Turing,\n\n"
                "I am writing to propose a potential research collaboration under the "
                "ResearchSphere AI umbrella. \n\n"
                "We identified high similarity between your medical computer vision models "
                "and Dr. Turing's self-explaining transformer layers. Combining these fields "
                "could yield significant advancements in trusted diagnostic intelligence.\n\n"
                "Would you be open to a 15-minute sync next Tuesday at 14:00 UTC?\n\n"
                "Sincerely,\nResearchSphere Orchestrator"
            )
            st.info("🔄 Approval state reset.")
            st.rerun()

# ------------------------------------------------------------------------------
# PAGE H: ⚙️ Settings
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "⚙️ Settings":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">⚙️ System Configuration</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Modify similarity search parameters, model assignments, and multi-agent settings.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model & Vector Index Controls")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.selectbox("Orchestration LLM", ["Gemini 2.5 Flash", "Gemini 1.5 Pro"], index=0)
        st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        st.checkbox("Enable SentenceTransformer Re-Ranking", value=True)
    with col_s2:
        st.text_input("Tavily API Search Bounds", value="site:arxiv.org OR site:scholar.google.com")
        st.number_input("ChromaDB Vector Retrieval Top-K Matches", min_value=1, max_value=10, value=3)
        st.checkbox("Require Human-in-the-Loop Email Verification", value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Environment Variables Verified Status")

    backend_status = _get_backend_status()
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size:0.85rem;">
            <div><b>GEMINI_API_KEY:</b> <span style="color:{'#10B981' if backend_status['gemini_configured'] else '#EF4444'}; font-weight:600;">{'✓ Configured' if backend_status['gemini_configured'] else '✗ Missing'}</span></div>
            <div><b>TAVILY_API_KEY:</b> <span style="color:{'#10B981' if backend_status['tavily_configured'] else '#EF4444'}; font-weight:600;">{'✓ Configured' if backend_status['tavily_configured'] else '✗ Missing'}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with env_col2:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size:0.85rem;">
            <div><b>CHROMA_STORE_DIR:</b> <code>{os.getenv('CHROMA_STORE_DIR', 'chroma_store')}</code></div>
            <div><b>FACULTY_DATASET_DIR:</b> <code>{os.path.join('data', 'faculty')}</code></div>
            <div><b>Chroma configured:</b> <span style="color:{'#10B981' if backend_status['chroma_configured'] else '#EF4444'}; font-weight:600;">{'✓ Yes' if backend_status['chroma_configured'] else '✗ No'}</span></div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# PAGE I: ℹ️ About
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "ℹ️ About":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: #0F172A; font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem 0;">ℹ️ About ResearchSphere AI</h2>
        <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Multi-Agent Orchestration framework mapping academic domains and facilitating supervisory matches.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card">
        <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:700;">Multi-Agent Architecture Blueprint</h4>
        <p style="font-size: 0.9rem; color: #475569; line-height: 1.6;">
            ResearchSphere AI operates as an event-driven multi-agent system managed under <b>LangGraph</b>.
            By routing user inputs through specialized specialists, the platform compiles publications,
            identifies market research trends, highlights open problems (gaps), and offers cohesive project proposals:
        </p>
        <ul style="font-size: 0.9rem; color: #475569; line-height: 1.6; margin-bottom: 1.5rem;">
            <li><b>StudentAgent:</b> Analyzes interest parameters, formatting queries for supervisor and publication mapping.</li>
            <li><b>FacultyRetrievalAgent:</b> Triggers vector similarity searches across locally ingested faculty profiles inside <b>ChromaDB</b> using <b>SentenceTransformer</b> embeddings.</li>
            <li><b>TrendAgent:</b> Synthesizes modern publication trajectories using real-time search queries and Google Gemini.</li>
            <li><b>GapAgent:</b> Compares trends against indexed internal faculty directions to locate academic voids.</li>
            <li><b>CollaborationAgent:</b> Calculates peer alignment scores to suggest cross-departmental co-authorships.</li>
            <li><b>ProjectRecommendationAgent:</b> Orchestrates supervisor, trend, and gap data to draft structured thesis recommendations.</li>
            <li><b>ConfirmationAgent:</b> Places high-consequence dispatches into the Human-in-the-Loop gateway queue.</li>
        </ul>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px; font-size: 0.85rem; color: #475569;">
            <b>Technical Stack:</b> Python 3.10 • Streamlit • LangGraph • SentenceTransformers (all-MiniLM-L6-v2) • ChromaDB Vector Database • Gemini 2.5 Flash SDK • Tavily Search API
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 7. FOOTER
# ==============================================================================
st.markdown("<hr style='margin: 3rem 0 1.25rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.75rem; font-weight: 500; letter-spacing: 0.025em;'>"
    "ResearchSphere AI &copy; 2026 | Powered by LangGraph &bull; Gemini &bull; ChromaDB &bull; Streamlit"
    "</div>",
    unsafe_allow_html=True
)
