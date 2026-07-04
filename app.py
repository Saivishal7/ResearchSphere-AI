# app.py
# ResearchSphere AI - Premium Streamlit UI Shell (Phase 1)
# Built with modern SaaS aesthetics, fluid layout, responsive components, and custom CSS

import streamlit as st
import datetime
import pandas as pd
import numpy as np

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

# ==============================================================================
# 3. PREMIUM THEME CSS INJECTION
# ==============================================================================
st.markdown("""
<style>
/* Premium Modern Custom CSS for ResearchSphere AI UI Shell */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main Page Styling overrides */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1E293B;
}

.stApp {
    background-color: #F8FAFC;
}

/* Hide default streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Custom containers padding and spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* Premium Card Design */
.custom-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    margin-bottom: 1.5rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.custom-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    border-color: #CBD5E1;
}

/* Metrics Dashboard Cards */
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 12px -3px rgba(0, 0, 0, 0.04);
    border-color: #CBD5E1;
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.metric-icon {
    font-size: 1.25rem;
    color: #3B82F6;
    background-color: #EFF6FF;
    padding: 0.375rem;
    border-radius: 8px;
    line-height: 1;
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0.25rem 0;
}

.metric-sub {
    font-size: 0.75rem;
    color: #10B981;
    font-weight: 500;
}

/* System Status badges */
.status-badge {
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 600;
    background-color: #F1F5F9;
    color: #64748B;
    border: 1px solid #E2E8F0;
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
}

.status-badge-dot {
    width: 6px;
    height: 6px;
    background-color: #94A3B8;
    border-radius: 50%;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    color: #F1F5F9;
}

[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: #94A3B8 !important;
}

/* Customize Streamlit Buttons to match SaaS styling */
div.stButton > button {
    background-color: #FFFFFF;
    color: #1E293B;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s ease;
    width: 100%;
}

div.stButton > button:hover {
    border-color: #3B82F6;
    color: #3B82F6;
    background-color: #F8FAFC;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}

div.stButton > button:active {
    transform: translateY(0);
}

/* Custom Notification and HIPLO boxes */
.hipl-alert {
    background-color: #FFFBEB;
    border-left: 4px solid #F59E0B;
    padding: 1.25rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. LEFT SIDEBAR (LOGO, ROLE SELECTOR, & NAVIGATION)
# ==============================================================================
with st.sidebar:
    # Title / Branding
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; margin-bottom: 0.25rem;">
        <span style="font-size: 2rem; line-height: 1;">🪐</span>
        <span style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.03em;">ResearchSphere</span>
    </div>
    <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 2rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Multi-Agent Orchestration</div>
    """, unsafe_allow_html=True)

    # User Role Selector
    st.markdown("<hr style='border-color: #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>USER PERSONA ROLE</div>", unsafe_allow_html=True)
    
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

    # Navigation Menu
    st.markdown("<hr style='border-color: #1E293B; margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>NAVIGATION</div>", unsafe_allow_html=True)
    
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
    
    selected_page = st.radio(
        "Navigation Menu",
        pages_list,
        index=pages_list.index(st.session_state.current_page) if st.session_state.current_page in pages_list else 0,
        label_visibility="collapsed",
        key="sidebar_nav_radio"
    )
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    # Muted details
    st.markdown("""
    <div style="position: fixed; bottom: 1.5rem; left: 1rem; font-size: 0.75rem; color: #475569;">
        <div><b>Node ID:</b> main-orchestrator</div>
        <div><b>Session State:</b> Local Storage</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. TOP HEADER
# ==============================================================================
current_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

st.markdown(f"""
<div class="custom-card" style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); color: white; border: none; padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 style="margin: 0; font-size: 1.75rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.025em; display: flex; align-items: center; gap: 0.5rem;">
                ResearchSphere AI
            </h1>
            <p style="margin: 0.25rem 0 0 0; color: #94A3B8; font-size: 0.9rem; font-weight: 400;">AI-Powered Research Collaboration Platform</p>
        </div>
        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
            <div style="text-align: right;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Current UTC Time</div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #F8FAFC; margin-top: 0.125rem;">{current_time_str}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Active Role</div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #3B82F6; margin-top: 0.125rem;">{st.session_state.role}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Backend Status</div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #94A3B8; display: flex; align-items: center; gap: 0.375rem; justify-content: flex-end; margin-top: 0.125rem;">
                    <span style="width: 7px; height: 7px; background-color: #64748B; border-radius: 50%;"></span>Unknown (Phase 1)
                </div>
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
    # 1. Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">Faculty Profiles</span>
                <span class="metric-icon">📁</span>
            </div>
            <div class="metric-value">15</div>
            <span class="metric-sub">✓ Fully Indexed</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">Research Areas</span>
                <span class="metric-icon">🏷️</span>
            </div>
            <div class="metric-value">12</div>
            <span class="metric-sub">✓ Domains Segmented</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">AI Agents</span>
                <span class="metric-icon">🤖</span>
            </div>
            <div class="metric-value">9</div>
            <span class="metric-sub">✓ Active Specializations</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">System Status</span>
                <span class="metric-icon">⚡</span>
            </div>
            <div class="metric-value" style="color:#64748B;">Standby</div>
            <span class="metric-sub" style="color:#64748B;">Phase 1 Frontend Shell</span>
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

        st.markdown("""
        <div class="custom-card" style="padding: 1.25rem;">
            <div style="display: flex; flex-direction: column; gap: 0.85rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">Gemini Service</span>
                    <span class="status-badge"><span class="status-badge-dot"></span>Unknown</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">ChromaDB Database</span>
                    <span class="status-badge"><span class="status-badge-dot"></span>Unknown</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">LangGraph Orchestrator</span>
                    <span class="status-badge"><span class="status-badge-dot"></span>Unknown</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.6rem; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">Tavily Search Engine</span>
                    <span class="status-badge"><span class="status-badge-dot"></span>Unknown</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500; font-size: 0.85rem; color: #475569;">Faculty Dataset</span>
                    <span class="status-badge"><span class="status-badge-dot"></span>Unknown</span>
                </div>
            </div>
            <div style="margin-top: 1.25rem; background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 8px; font-size: 0.75rem; color: #64748B; line-height: 1.4;">
                💡 <b>Developer Note:</b> These connect during Phase 2. The Streamlit shell will seamlessly consume the <code>build_graph.py</code> and <code>gemini_service.py</code> backends.
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
            submit_btn = st.form_submit_form_button = st.form_submit_button("🔍 Find Matches")
        
        if submit_btn:
            st.session_state.current_query = student_interest
            st.warning("⚠️ Phase 1 Sandbox: Real model routing and ChromaDB retrieval will activate during the Phase 2 integration.")

    # Show Illustrative Workflow & Mockup Alignment
    if st.session_state.current_query:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🗺️ Orchestrated Matchmaker Output (Phase 1 Blueprint)")
        
        col_out1, col_out2 = st.columns([1, 1])
        with col_out1:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Matched Supervisor Match (ChromaDB Vector Mockup)</h4>
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                    <div>
                        <span style="font-weight: 700; font-size: 1.1rem; color: #0F172A;">Dr. Sarah Jenkins</span><br>
                        <span style="font-size: 0.8rem; color: #64748B;">Department of Computer Science</span>
                    </div>
                    <span style="background-color: #E0F2FE; color: #0369A1; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.5rem; border-radius: 4px;">92% Match Score</span>
                </div>
                <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 0.75rem;">
                    <b>Primary Areas:</b> Medical Computer Vision, Explainable Neural Networks, Attention Mechanisms in ViTs.
                </p>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem; border-radius: 6px; font-size: 0.8rem; color: #475569;">
                    <b>Relevant Publication:</b> "Self-Explaining Attention Maps for Multi-Modal Breast MRI Classification", CVPR 2025.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_out2:
            st.markdown("""
            <div class="custom-card">
                <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Orchestrated Research Direction</h4>
                <div style="font-weight: 700; color: #0F172A; font-size: 1.05rem; margin-bottom: 0.5rem;">
                    Self-Explaining Attentive MRI Segmentations
                </div>
                <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 1rem;">
                    Integrating explainable features directly into Vision Transformer (ViT) bottleneck layers to create transparent annotations for oncology radiologists.
                </p>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <span style="background-color: #F1F5F9; color: #475569; font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px;">LangGraph Routed</span>
                    <span style="background-color: #F1F5F9; color: #475569; font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px;">FacultyRetrievalAgent</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
        value="Explainable Neural Networks"
    )

    col_prof1, col_prof2, col_prof3 = st.columns(3)
    with col_prof1:
        trend_trigger = st.button("📈 Extract Research Trends")
    with col_prof2:
        gap_trigger = st.button("🎯 Identify Research Gaps")
    with col_prof3:
        collab_trigger = st.button("🤝 Discover Potential Co-Authors")

    if trend_trigger or gap_trigger or collab_trigger:
        st.warning("⚠️ Phase 1 Frontend Sandbox: The underlying agents (TrendAgent, GapAgent, and CollaborationAgent) will run in Phase 2.")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📋 Sandbox Output Representation")
        
        if trend_trigger:
            st.markdown("""
            <div class="custom-card">
                <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Emerging Vectors (TrendAgent Output Illustration)</h4>
                <ul style="font-size: 0.85rem; color: #475569; line-height: 1.6; margin-bottom: 0;">
                    <li><b>Trend A:</b> Rapid adoption of Concept Bottleneck Models (CBM) as alternatives to post-hoc explainers (+42% YoY citation).</li>
                    <li><b>Trend B:</b> Self-explaining transformer heads configured natively in foundational LLM backbones.</li>
                    <li><b>Trend C:</b> Interactive human-in-the-loop explanation refinement workflows.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif gap_trigger:
            st.markdown("""
            <div class="custom-card" style="border-left: 4px solid #EF4444;">
                <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Niche Academic Gaps (GapAgent Output Illustration)</h4>
                <ul style="font-size: 0.85rem; color: #475569; line-height: 1.6; margin-bottom: 0;">
                    <li><b>Identified Gap:</b> Absence of generalized benchmark suites evaluating explanation consistency across highly multi-modal diagnostic tasks.</li>
                    <li><b>Niche Intersection:</b> Applying adversarial training directly on concept layers to prevent explainability spoofing attacks.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif collab_trigger:
            st.markdown("""
            <div class="custom-card" style="border-left: 4px solid #10B981;">
                <h4 style="margin: 0 0 0.75rem 0; color: #1E293B; font-weight:600;">Complementary Co-Authorship Pairing (CollaborationAgent Output)</h4>
                <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 0.5rem;">
                    <b>Recommended Match:</b> <b>Dr. Sarah Jenkins</b> (Medical Vision expert) with <b>Dr. Alan Turing</b> (NLP Theory/Explainability expert).
                </p>
                <p style="font-size: 0.85rem; color: #64748B; line-height: 1.5; margin: 0;">
                    <b>Rationale:</b> Jenkins possesses rich clinical visual data; Turing has the algorithmic transformer bottlenecks. Merging these creates state-of-the-art diagnostic explainability models.
                </p>
            </div>
            """, unsafe_allow_html=True)

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

    # Render a premium mockup recommendation cards bento
    st.markdown("### Generated Thesis Proposals (Demo Catalog)")
    
    st.markdown("""
    <div class="custom-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <h3 style="margin:0 0 0.25rem 0; font-size:1.2rem; color:#0F172A; font-weight:700;">A Transparent Bottleneck Layer for Vision Transformer Architectures</h3>
                <span style="font-size: 0.8rem; color: #64748B;">Target Area: Medical computer vision + explainability validation</span>
            </div>
            <span style="background-color: #F1F5F9; color: #475569; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.5rem; border-radius: 4px;">Supervisor Match: Dr. Sarah Jenkins</span>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem; margin-bottom: 1rem;">
            <div>
                <h5 style="margin:0 0 0.375rem 0; font-weight:600; color:#475569; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.02em;">Problem Statement & Gap</h5>
                <p style="font-size:0.85rem; color:#1E293B; line-height:1.5; margin:0;">
                    Post-hoc explainers like Grad-CAM are often inconsistent and can spoof visual cues, creating critical trust barriers for clinical oncologists reviewing diagnostic maps.
                </p>
            </div>
            <div>
                <h5 style="margin:0 0 0.375rem 0; font-weight:600; color:#475569; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.02em;">Orchestrated Methodology</h5>
                <p style="font-size:0.85rem; color:#1E293B; line-height:1.5; margin:0;">
                    Deploying self-explaining Concept Bottleneck Layers directly into the feedforward blocks of Vision Transformers to output explicit, human-comprehensible bounding boxes.
                </p>
            </div>
        </div>
        
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.85rem; border-radius: 8px; font-size: 0.8rem; color: #475569; line-height: 1.4;">
            <b>Generated References:</b><br>
            [1] Jenkins, S. "Self-Explaining Attention Maps", CVPR 2025.<br>
            [2] Turing, A. "Bottleneck Layer Concepts in Deep Transformers", Journal of AI Theory, 2024.
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    # 1. Action Notification Box
    st.markdown(f"""
    <div class="hipl-alert">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <b style="color: #92400E; font-size: 1.1rem;">⚠️ Verification Required: {st.session_state.approval_state['action']}</b>
            <span style="background-color: #FEF3C7; color: #B45309; border: 1px solid #F59E0B; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">
                {st.session_state.approval_state['status'].upper()}
            </span>
        </div>
        <div style="color: #78350F; font-size: 0.85rem; line-height: 1.5;">
            <b>Justification:</b> {st.session_state.approval_state['reason']}<br>
            <b>Target Faculty Node Pair:</b> {', '.join(st.session_state.approval_state['affected_faculty'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Text Editor for generated content
    edited_text = st.text_area(
        "Edit Draft Communication / Command dispatch:",
        value=st.session_state.approval_state["generated_content"],
        height=240,
        disabled=(st.session_state.approval_state["status"] != "Pending Verification")
    )
    
    # Save the updated content in session_state
    if edited_text != st.session_state.approval_state["generated_content"]:
        st.session_state.approval_state["generated_content"] = edited_text

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 3. Decision Buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("👍 APPROVE & AUTHORIZE DISPATCH", disabled=(st.session_state.approval_state["status"] != "Pending Verification")):
            st.session_state.approval_state["status"] = "Approved & Sent"
            st.success("✅ Action authorized! Under real operations, this triggers the Email Dispatch Tool / Webhook pipeline.")
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
            st.info("🔄 Sandbox approval state reset.")
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

    # Sandbox configurations
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
    
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size:0.85rem;">
            <div><b>GEMINI_API_KEY:</b> <span style="color:#10B981; font-weight:600;">✓ Configured</span></div>
            <div><b>TAVILY_API_KEY:</b> <span style="color:#10B981; font-weight:600;">✓ Configured</span></div>
        </div>
        """, unsafe_allow_html=True)
    with env_col2:
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size:0.85rem;">
            <div><b>CHROMA_PERSIST_DIR:</b> <code>./data/chroma</code></div>
            <div><b>FACULTY_DATASET_DIR:</b> <code>./data/faculty</code></div>
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
