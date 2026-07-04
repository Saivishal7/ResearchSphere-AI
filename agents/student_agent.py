# agents/student_agent.py
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from graph.state import ResearchState
from tools.gemini_service import GeminiService
from agents.faculty_retrieval_agent import FacultyRetrievalAgent, FacultyMatch, FacultyRetrievalResult

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StudentAgentResponse(BaseModel):
    """
    Standardized, production-quality response model representing structured supervisor recommendations.
    Provides complete type safety and shields future AI agents from inconsistent string formatting.
    """
    query: str
    status: str = Field(default="success", description="Execution status: success, partial, or failed.")
    recommended_faculty: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked list of recommended faculty supervisors with reasons and strengths summary."
    )
    reasoning: str = Field(
        ...,
        description="Overall comparative reasoning, ranking breakdown, and best supervisor selection."
    )
    alternative_faculty: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Secondary faculty matches that align with the query but are lower rank."
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested insightful questions for the student to consider or ask during outreach."
    )


class StudentAgent:
    """
    The Student Agent orchestrator. Responsible for processing student queries,
    determining metadata filters, invoking the Faculty Retrieval Agent (RAG),
    and leveraging the Gemini Service for synthesis, ranking, and justification.
    
    This agent strictly enforces NO direct database connections and NO hallucinations,
    limiting Gemini's reasoning entirely to retrieved context.
    """
    def __init__(self):
        self.gemini = GeminiService()
        self.retriever = FacultyRetrievalAgent()

    def _extract_filters(self, query: str) -> Dict[str, Optional[str]]:
        """
        Interrogates GeminiService to extract optional metadata filters (department, capacity_status)
        from the user's natural query.
        """
        default_filters = {"department": None, "capacity_status": None}
        
        # If Gemini is not healthy or available, skip filter extraction gracefully
        if not self.gemini.health_check():
            logger.warning("GeminiService health check failed. Skipping query filter extraction.")
            return default_filters

        try:
            prompt = f"""You are an advanced academic assistant parsing student inquiries.
Extract optional metadata filters from the student query.

FILTERS TO EXTRACT:
1. "department": Match a department name exactly (e.g., "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Biomedical Engineering") or null if no department is specified.
2. "capacity_status": Must be "accepting" if they ask for supervisors who are "accepting", "have vacancy", "taking students", or "have capacity". Must be "full" if they ask for "not taking", "full", "no capacity". Otherwise, return null.

STUDENT QUERY:
"{query}"

Respond ONLY with a valid JSON object containing keys:
{{
  "department": string or null,
  "capacity_status": string or null
}}"""
            logger.info("Extracting optional filters from student query using Gemini...")
            result = self.gemini.generate_json(prompt)
            
            if isinstance(result, dict):
                department = result.get("department")
                capacity_status = result.get("capacity_status")
                logger.info(f"Extracted Filters -> Department: {department}, Capacity: {capacity_status}")
                return {
                    "department": department if isinstance(department, str) else None,
                    "capacity_status": capacity_status if isinstance(capacity_status, str) else None
                }
            return default_filters
        except Exception as e:
            logger.warning(f"Failed to extract query filters via Gemini (falling back to none): {e}")
            return default_filters

    def search_supervisors(self, query: str) -> StudentAgentResponse:
        """
        Orchestrates the entire student advising workflow:
        1. Extract query metadata filters via GeminiService.
        2. Retrieve relevant faculty profiles via FacultyRetrievalAgent (ChromaDB).
        3. Handle empty results gracefully.
        4. Synthesize, rank, and explain matches via GeminiService using ONLY the retrieved context.
        """
        if not query or not query.strip():
            return StudentAgentResponse(
                query="",
                status="failed",
                recommended_faculty=[],
                reasoning="Empty query provided. Please ask a valid question about research topics or faculty.",
                alternative_faculty=[],
                follow_up_questions=["What research areas or professors are you interested in?"]
            )

        # 1. Determine optional filters
        filters = self._extract_filters(query)
        department = filters.get("department")
        capacity_status = filters.get("capacity_status")

        # 2. Call FacultyRetrievalAgent
        logger.info(f"Invoking FacultyRetrievalAgent with query: '{query}'")
        retrieval_result: FacultyRetrievalResult = self.retriever.retrieve_relevant_faculty(
            query=query,
            k=5,
            department=department,
            capacity_status=capacity_status
        )

        # 3. Handle empty retrieval / errors gracefully
        if not retrieval_result.success or not retrieval_result.matches:
            msg = retrieval_result.error_message or "No matching faculty supervisors found in our database."
            logger.info(f"No faculty matches returned: {msg}")
            return StudentAgentResponse(
                query=query,
                status="partial",
                recommended_faculty=[],
                reasoning=f"No faculty matches were found for your query. Reason: {msg}",
                alternative_faculty=[],
                follow_up_questions=[
                    "Would you like to search across all departments instead?",
                    "Try using broader keywords (e.g., 'artificial intelligence' instead of 'Healthcare AI')",
                    "Are you looking for supervisors who are accepting new students?"
                ]
            )

        matches = retrieval_result.matches
        logger.info(f"Successfully retrieved {len(matches)} faculty matches from database.")

        # 4. Build concise, rich context from retrieved faculty
        context_blocks = []
        for match in matches:
            interests_str = ", ".join(match.research_interests)
            keywords_str = ", ".join(match.keywords)
            
            projects_list = []
            for p in match.current_projects:
                projects_list.append(f"  - Title: {p.title}\n    Description: {p.description}")
            projects_str = "\n".join(projects_list)
            
            publications_list = []
            for pub in match.publications:
                publications_list.append(f"  - Title: {pub.title} ({pub.year})\n    Venue: {pub.conference_or_journal}")
            publications_str = "\n".join(publications_list)

            block = f"""Faculty ID: {match.faculty_id}
Name: {match.name}
Designation: {match.designation}
Department: {match.department}
Capacity Status: {match.capacity_status}
Research Interests: {interests_str}
Profile Summary: {match.profile_summary}
Current Projects:
{projects_str}
Key Publications:
{publications_str}
Keywords: {keywords_str}
Similarity Score: {match.similarity_score:.4f}
---"""
            context_blocks.append(block)

        retrieved_context = "\n".join(context_blocks)

        # 5. Formulate prompt for Gemini with strict zero-hallucination rules
        prompt = f"""You are an advanced academic advising agent.
Your goal is to help a student select the best faculty supervisor based ONLY on the provided context.

STUDENT QUERY:
"{query}"

RETIRIED FACULTY CONTEXT:
{retrieved_context}

STRICT OPERATIONAL RULES:
1. Answer the query relying ONLY on the retrieved faculty context above. Do NOT assume, extrapolate, or hallucinate credentials, projects, or publications not mentioned.
2. If the context does not contain enough information to answer a specific question, explicitly state that the information is unavailable in the retrieved context.
3. Determine the best matching supervisor and explain why they are recommended.
4. Provide structured, clear explanation of matches for both recommended and alternative faculty.
5. Rank suitability starting with rank 1 as the highest.

You must respond strictly with a valid JSON object matching the schema below (Do not include markdown blocks or wrappers other than the raw JSON itself):
{{
  "query": "string (original student query)",
  "recommended_faculty": [
    {{
      "faculty_id": "string",
      "name": "string",
      "department": "string",
      "designation": "string",
      "match_explanation": "string (detailed justification of how their research/projects match the student query based ONLY on context)",
      "rank": 1,
      "strengths_summary": "string (bulleted or concise summary of key strengths)"
    }}
  ],
  "reasoning": "string (overall comparative reasoning, ranking methodology, and clear justification of who the primary recommended supervisor is and why)",
  "alternative_faculty": [
    {{
      "faculty_id": "string",
      "name": "string",
      "department": "string",
      "designation": "string",
      "match_explanation": "string (why they serve as a good secondary or alternative matching supervisor based ONLY on context)"
    }}
  ],
  "follow_up_questions": [
    "string (insightful, specific follow-up questions the student could ask these faculty members)"
  ]
}}"""

        # 6. Call Gemini Service
        try:
            logger.info("Sending retrieved context and query to GeminiService...")
            res_json = self.gemini.generate_json(prompt)
            
            if not isinstance(res_json, dict):
                raise ValueError("Response from GeminiService was not a structured dictionary.")

            # Parse defensively to guarantee perfect type structure
            query_val = res_json.get("query", query)
            recommended = res_json.get("recommended_faculty", [])
            reasoning_val = res_json.get("reasoning", "Overall matching evaluation completed.")
            alternatives = res_json.get("alternative_faculty", [])
            follow_ups = res_json.get("follow_up_questions", [])

            if not isinstance(recommended, list):
                recommended = []
            if not isinstance(alternatives, list):
                alternatives = []
            if not isinstance(follow_ups, list):
                follow_ups = []

            return StudentAgentResponse(
                query=query_val,
                status="success",
                recommended_faculty=recommended,
                reasoning=reasoning_val,
                alternative_faculty=alternatives,
                follow_up_questions=follow_ups
            )
        except Exception as e:
            logger.error(f"Structured response generation failed: {e}. Falling back to defensive formatting.")
            # Highly stable defensive fallback mapping
            return StudentAgentResponse(
                query=query,
                status="partial",
                recommended_faculty=[
                    {
                        "faculty_id": m.faculty_id,
                        "name": m.name,
                        "department": m.department,
                        "designation": m.designation,
                        "match_explanation": f"Statically aligned with interests: {', '.join(m.research_interests)}.",
                        "rank": i + 1,
                        "strengths_summary": m.profile_summary[:150] + "..."
                    } for i, m in enumerate(matches[:2])
                ],
                reasoning=f"Faculty matches were successfully retrieved via database semantic similarity. Synthesis reasoning failed due to: {e}.",
                alternative_faculty=[
                    {
                        "faculty_id": m.faculty_id,
                        "name": m.name,
                        "department": m.department,
                        "designation": m.designation,
                        "match_explanation": "Alternative match based on similarity score."
                    } for m in matches[2:]
                ],
                follow_up_questions=[
                    "Would you like to search for related publications on this topic?",
                    "Are you interested in viewing the contact information for these faculty members?"
                ]
            )


def student_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph agent node responsible for analyzing student research interests and recommending faculty.
    Orchestrates the StudentAgent to perform retrieval, ranking, and explanation.
    """
    print("[Agent Node] executing student_agent_node...")
    query = state.get("current_query", "")
    
    agent = StudentAgent()
    try:
        response = agent.search_supervisors(query)
        # Flatten matches for state storage
        all_matches_flat = []
        for r in response.recommended_faculty:
            all_matches_flat.append(r)
        for alt in response.alternative_faculty:
            all_matches_flat.append(alt)

        return {
            "faculty_results": all_matches_flat,
            "retrieved_context": response.reasoning,
            "error": None
        }
    except Exception as e:
        logger.error(f"Student Agent node execution error: {e}")
        return {
            "faculty_results": [],
            "retrieved_context": None,
            "error": f"StudentAgentNode failed: {str(e)}"
        }
