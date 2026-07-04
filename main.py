# main.py
# Production CLI Entrypoint for ResearchSphere AI

import sys
import json
from config import validate_config
from graph.build_graph import compiled_research_graph
from graph.state import ResearchState

def run_query(query: str) -> None:
    """
    Initializes the LangGraph state, executes the query through the compiled graph,
    and prints highly structured, human-readable output of the final research state.
    """
    print("\n" + "="*80)
    print(f" RESEARCHSPHERE AI - GRAPH EXECUTION")
    print("="*80)
    print(f"User Query: '{query}'")
    print("-" * 80)

    # 1. Initialize complete ResearchState with defaults
    initial_state: ResearchState = {
        "current_query": query,
        "user_role": "unknown",
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
        "session_id": "cli-session-001"
    }

    try:
        # 2. Invoke the compiled LangGraph workflow
        final_state = compiled_research_graph.invoke(initial_state)

        # 3. Print Structured Outputs beautifully
        print(f"Identified Role:   {final_state.get('user_role', 'unknown').upper()}")
        print(f"Identified Intent: {final_state.get('intent', 'general_query').upper()}")
        print(f"Execution Error:   {final_state.get('error') or 'None'}")
        print("-" * 80)

        # Case A: Student Advising Faculty Results
        if final_state.get("faculty_results"):
            print("\n🔍 STRUCTURED FACULTY SUPERVISOR RESULTS:")
            results = final_state["faculty_results"]
            print(json.dumps(results, indent=2))

        # Case B: Professor Research Trends
        if final_state.get("research_trends"):
            print("\n📈 STRUCTURED RESEARCH TRENDS ANALYSIS:")
            print(json.dumps(final_state["research_trends"], indent=2))

        # Case C: Professor Research Gaps
        if final_state.get("research_gaps"):
            print("\n🔬 STRUCTURED RESEARCH GAPS ANALYSIS:")
            print(json.dumps(final_state["research_gaps"], indent=2))

        # Case D: Professor Collaboration Suggestions
        if final_state.get("collaboration_suggestions"):
            print("\n🤝 STRUCTURED COLLABORATION SUGGESTIONS:")
            print(json.dumps(final_state["collaboration_suggestions"], indent=2))

        # Case E: Student/Professor Project Recommendations
        if final_state.get("project_recommendations"):
            print("\n💡 STRUCTURED PROJECT RECOMMENDATIONS:")
            print(json.dumps(final_state["project_recommendations"], indent=2))

        # Case F: Human-in-the-Loop Confirmation Pending
        if final_state.get("approval_required"):
            print("\n⚠️ HUMAN-IN-THE-LOOP APPROVAL REQUIRED:")
            print("The system has paused a high-consequence action awaiting human consent.")
            if final_state.get("pending_action"):
                print(json.dumps(final_state["pending_action"], indent=2))

        # Printed Summary/Contextual Synthesis
        if final_state.get("retrieved_context"):
            print("\n📖 SYNTHESIS & RECOMMENDATIONS SUMMARY:")
            print(final_state["retrieved_context"])

    except Exception as e:
        print(f"\n❌ GRAPH EXECUTION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

    print("=" * 80 + "\n")


def main():
    # 1. Validate environment configuration
    is_valid, warnings = validate_config()
    if not is_valid:
        print("[WARNING] Configuration Warnings Found:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # 2. Extract query from CLI arguments or use standard default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default query for demonstrating a complete execution path
        query = "What are the emerging trends in Explainable Artificial Intelligence?"

    # 3. Run the complete pipeline
    run_query(query)


if __name__ == "__main__":
    main()
