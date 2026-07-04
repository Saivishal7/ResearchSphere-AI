from typing import TypedDict, Optional, List, Dict, Any, Literal

class ResearchState(TypedDict):
    """
    State definition for the ResearchSphere AI multi-agent system.
    This TypedDict serves as the single source of truth for the LangGraph state,
    propagating user intent, profiles, retrieved contexts, and routing parameters.
    """
    # Active input query from user
    current_query: str

    # Identity and Role of the user
    user_role: Literal["student", "professor", "unknown"]

    # Identified research intent
    intent: Literal[
        "faculty_search",
        "faculty_details",
        "research_trends",
        "research_gap",
        "collaboration",
        "project_recommendation",
        "general_query",
        "clarification"
    ]

    # Append-only list of conversation turns: [{"role": "user"|"ai", "text": "..."}]
    conversation_history: List[Dict[str, str]]

    # Profile representations for context building
    student_profile: Optional[Dict[str, Any]]
    professor_profile: Optional[Dict[str, Any]]

    # Results from specialized actions
    faculty_results: Optional[List[Dict[str, Any]]]
    selected_faculty: Optional[Dict[str, Any]]
    research_trends: Optional[Dict[str, Any]]
    research_gaps: Optional[Dict[str, Any]]
    collaboration_suggestions: Optional[List[Dict[str, Any]]]
    project_recommendations: Optional[List[Dict[str, Any]]]

    # Human-in-the-loop action validation parameters
    pending_action: Optional[Dict[str, Any]]
    approval_required: bool
    approval_status: Literal["pending", "approved", "modified", "rejected"]

    # Contextual fields for pipeline state
    retrieved_context: Optional[str]
    tool_output: Optional[str]
    error: Optional[str]
    retry_count: int
    session_id: str
