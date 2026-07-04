from typing import Dict, Any, Literal
from graph.state import ResearchState

def classify_intent(query: str) -> Literal[
    "faculty_search",
    "faculty_details",
    "research_trends",
    "research_gap",
    "collaboration",
    "project_recommendation",
    "general_query",
    "clarification"
]:
    """
    Classifies a user query into a research intent using keyword-based heuristics.
    This function is decoupled from the LangGraph node logic to make it modular and 
    easy to swap out for an LLM/Gemini router in later steps.
    """
    q = query.lower().strip()
    
    # 1. Project Recommendation
    if any(k in q for k in ["project idea", "proposal", "recommend direction", "thesis topic", "suggest a project", "project recommendation"]):
        return "project_recommendation"
        
    # 2. Research Gaps
    if any(k in q for k in ["gap", "unexplored", "under-explored", "sparse coverage", "research gap"]):
        return "research_gap"
        
    # 3. Collaboration
    if any(k in q for k in ["collaborator", "collaborate", "cross-department", "co-author", "partnership", "work with"]):
        return "collaboration"
        
    # 4. Research Trends
    if any(k in q for k in ["trend", "emerging", "latest in", "hot topics", "state of the art", "whats new"]):
        return "research_trends"
        
    # 5. Faculty Details
    if any(k in q for k in ["details on", "profile of", "who is", "tell me about dr", "background on"]):
        return "faculty_details"
        
    # 6. Faculty Search
    if any(k in q for k in ["search", "find", "supervisor", "advisor", "faculty", "looking for", "match"]):
        return "faculty_search"
        
    # 7. Clarification / Hesitation
    if any(k in q for k in ["not sure", "unclear", "explain", "how do i", "can you clarify"]):
        return "clarification"
        
    # 8. General / Fallback
    return "general_query"


def router_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node that reads the current user query, determines the intent,
    and updates the state context accordingly.
    """
    query = state.get("current_query", "")
    intent = classify_intent(query)
    
    # Simple heuristic to identify roles from intents
    current_role = state.get("user_role", "unknown")
    if current_role == "unknown" or not current_role:
        if intent in ["faculty_search", "faculty_details"]:
            user_role = "student"
        elif intent in ["research_trends", "research_gap", "collaboration"]:
            user_role = "professor"
        else:
            user_role = "unknown"
    else:
        user_role = current_role
        
    return {
        "intent": intent,
        "user_role": user_role
    }
