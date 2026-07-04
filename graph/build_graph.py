from langgraph.graph import StateGraph, START, END
from graph.state import ResearchState
from graph.router import router_node

# Import agent stub nodes
from agents.student_agent import student_agent_node
from agents.professor_agent import professor_agent_node
from agents.faculty_retrieval_agent import faculty_retrieval_agent_node
from agents.trend_agent import trend_agent_node
from agents.gap_agent import gap_agent_node
from agents.collaboration_agent import collaboration_agent_node
from agents.project_recommendation_agent import project_recommendation_agent_node
from agents.confirmation_agent import confirmation_agent_node

# Define routing conditionals
def route_intent(state: ResearchState) -> str:
    """
    Evaluates the intent computed by the router and maps it to the
    initial workspace coordinator (student vs professor flows).
    """
    intent = state.get("intent", "general_query")
    
    # Student workflow intents (including project_recommendation)
    if intent in ["faculty_search", "faculty_details", "clarification", "project_recommendation", "general_query"]:
        return "student_agent"
    # Professor workflow intents
    elif intent in ["research_trends", "research_gap", "collaboration"]:
        return "professor_agent"
    else:
        return "student_agent"

def route_student_flow(state: ResearchState) -> str:
    """
    Directs StudentAgent to FacultyRetrievalAgent.
    """
    return "faculty_retrieval_agent"

def route_faculty_retrieval(state: ResearchState) -> str:
    """
    Routes from FacultyRetrievalAgent to ProjectRecommendationAgent (if requested),
    or to ConfirmationAgent (if required), or END.
    """
    intent = state.get("intent")
    if intent == "project_recommendation":
        return "project_recommendation_agent"
    
    if state.get("approval_required", False):
        return "confirmation_agent"
        
    return "END"

def route_project_recommendation(state: ResearchState) -> str:
    """
    Routes from ProjectRecommendationAgent to ConfirmationAgent (if required) or END.
    """
    if state.get("approval_required", False):
        return "confirmation_agent"
    return "END"

def route_professor_flow(state: ResearchState) -> str:
    """
    Forks the professor workflow into specific analytical agents based on intent.
    """
    intent = state.get("intent")
    if intent == "research_gap":
        return "gap_agent"
    elif intent == "collaboration":
        return "collaboration_agent"
    else:
        return "trend_agent"

def route_trend_flow(state: ResearchState) -> str:
    """
    Routes from TrendAgent to ConfirmationAgent (if required) or END.
    """
    if state.get("approval_required", False):
        return "confirmation_agent"
    return "END"

def route_gap_flow(state: ResearchState) -> str:
    """
    Routes from GapAnalysisAgent to ConfirmationAgent (if required) or END.
    """
    if state.get("approval_required", False):
        return "confirmation_agent"
    return "END"

def route_collaboration_flow(state: ResearchState) -> str:
    """
    Routes from CollaborationAgent to ConfirmationAgent (if required) or END.
    """
    if state.get("approval_required", False):
        return "confirmation_agent"
    return "END"


def build_research_graph() -> StateGraph:
    """
    Builds and wires the multi-agent StateGraph.
    """
    # 1. Instantiate the graph with our state schema
    workflow = StateGraph(ResearchState)

    # 2. Add all 9 distinct processing nodes
    workflow.add_node("router", router_node)
    workflow.add_node("student_agent", student_agent_node)
    workflow.add_node("faculty_retrieval_agent", faculty_retrieval_agent_node)
    workflow.add_node("professor_agent", professor_agent_node)
    workflow.add_node("trend_agent", trend_agent_node)
    workflow.add_node("gap_agent", gap_agent_node)
    workflow.add_node("collaboration_agent", collaboration_agent_node)
    workflow.add_node("project_recommendation_agent", project_recommendation_agent_node)
    workflow.add_node("confirmation_agent", confirmation_agent_node)

    # 3. Configure entry path
    workflow.add_edge(START, "router")

    # 4. Wire router conditional pathways
    workflow.add_conditional_edges(
        "router",
        route_intent,
        {
            "student_agent": "student_agent",
            "professor_agent": "professor_agent"
        }
    )

    # 5. Wire student agent conditional pathways
    workflow.add_conditional_edges(
        "student_agent",
        route_student_flow,
        {
            "faculty_retrieval_agent": "faculty_retrieval_agent"
        }
    )

    # 6. Wire professor agent conditional pathways
    workflow.add_conditional_edges(
        "professor_agent",
        route_professor_flow,
        {
            "trend_agent": "trend_agent",
            "gap_agent": "gap_agent",
            "collaboration_agent": "collaboration_agent"
        }
    )

    # 7. Wire trend agent conditional pathways
    workflow.add_conditional_edges(
        "trend_agent",
        route_trend_flow,
        {
            "confirmation_agent": "confirmation_agent",
            "END": END
        }
    )

    # 8. Wire gap agent conditional pathways
    workflow.add_conditional_edges(
        "gap_agent",
        route_gap_flow,
        {
            "confirmation_agent": "confirmation_agent",
            "END": END
        }
    )

    # 9. Wire collaboration agent conditional pathways
    workflow.add_conditional_edges(
        "collaboration_agent",
        route_collaboration_flow,
        {
            "confirmation_agent": "confirmation_agent",
            "END": END
        }
    )

    # 10. Wire faculty retrieval and project recommendation pathways
    workflow.add_conditional_edges(
        "faculty_retrieval_agent",
        route_faculty_retrieval,
        {
            "project_recommendation_agent": "project_recommendation_agent",
            "confirmation_agent": "confirmation_agent",
            "END": END
        }
    )

    workflow.add_conditional_edges(
        "project_recommendation_agent",
        route_project_recommendation,
        {
            "confirmation_agent": "confirmation_agent",
            "END": END
        }
    )

    # 11. Connect confirmation static termination
    workflow.add_edge("confirmation_agent", END)

    return workflow


# Compile the graph for end-user execution
compiled_research_graph = build_research_graph().compile()
