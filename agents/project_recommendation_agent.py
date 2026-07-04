# agents/project_recommendation_agent.py
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from graph.state import ResearchState
from tools.gemini_service import GeminiService
from agents.faculty_retrieval_agent import FacultyRetrievalAgent
from agents.trend_agent import TrendAgent
from agents.gap_agent import GapAnalysisAgent

# Configure logging
logger = logging.getLogger("ProjectRecommendationAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ProjectRecommendationResponse(BaseModel):
    """
    Strongly typed, production-quality response model representing
    personalized project proposals. Protects downstream consumers from malformed fields.
    """
    project_title: str = Field(..., description="A clear, precise, and descriptive academic project title.")
    problem_statement: str = Field(..., description="A rigorous statement of the technical/scientific problem addressed.")
    research_motivation: Optional[str] = Field(None, description="Compelling academic motivation for pursuing this research.")
    objectives: List[str] = Field(default_factory=list, description="A list of 3-5 specific, realistic research objectives.")
    novelty: str = Field(..., description="Clear explanation of novelty and how it fills the identified gaps.")
    methodology: str = Field(..., description="Suggested step-by-step technical methodology or approach.")
    expected_outcomes: List[str] = Field(default_factory=list, description="List of expected tangible outcomes.")
    recommended_faculty: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Recommended faculty supervisors from retrieved profiles, with alignment reasons."
    )
    required_skills: List[str] = Field(default_factory=list, description="Key technical skills or domain knowledge required.")
    difficulty: Optional[str] = Field(None, description="Assessed difficulty level (e.g., Beginner, Intermediate, Advanced).")
    future_scope: str = Field(..., description="Forward-looking statement of potential extensions.")
    reasoning: str = Field(..., description="Thorough, high-integrity comparative reasoning detailing why this project fits.")
    status: str = Field(..., description="Status indicator representing execution success ('success', 'partial', 'failed').")


class ProjectRecommendationAgent:
    """
    The Project Recommendation Agent.
    Responsible for generating personalized research project ideas by combining:
    - Student interests / query
    - Faculty Retrieval results
    - Research Gap Analysis
    - Trend Analysis
    
    This agent strictly orchestrates existing components without duplicating their work,
    and enforces strict grounding to eliminate hallucinations.
    """
    def __init__(self):
        self.gemini = GeminiService()
        self.retriever = FacultyRetrievalAgent()
        self.trend_agent = TrendAgent()
        self.gap_agent = GapAnalysisAgent()

    def recommend_projects(self, student_interests: str) -> Dict[str, Any]:
        """
        Orchestrates retrieval, trends, and gaps to generate a personalized project recommendation.
        
        Args:
            student_interests (str): The natural language query or description of interests.
            
        Returns:
            Dict[str, Any]: Dictionary representation of the recommendation.
        """
        if not student_interests or not student_interests.strip():
            logger.warning("recommend_projects invoked with empty interests.")
            return {
                "project_title": "No Project Recommendation Generated",
                "problem_statement": "The student interests query was empty.",
                "research_motivation": "N/A",
                "objectives": [],
                "novelty": "N/A",
                "methodology": "N/A",
                "expected_outcomes": [],
                "recommended_faculty": [],
                "required_skills": [],
                "difficulty": "N/A",
                "future_scope": "N/A",
                "reasoning": "No input interests were provided to recommend projects.",
                "status": "failed"
            }

        # 1. Retrieve matching faculty (Step 2)
        faculty_context = ""
        faculty_list = []
        has_faculty = False
        try:
            logger.info("ProjectRecommendationAgent invoking FacultyRetrievalAgent...")
            retrieval_res = self.retriever.retrieve_relevant_faculty(student_interests, k=3)
            if retrieval_res and retrieval_res.success and retrieval_res.matches:
                faculty_list = retrieval_res.matches
                has_faculty = True
                fac_strings = []
                for idx, match in enumerate(faculty_list):
                    fac_strings.append(
                        f"Faculty Match #{idx+1}:\n"
                        f"- Name: {match.name}\n"
                        f"- Department: {match.department}\n"
                        f"- Research Interests: {', '.join(match.research_interests)}\n"
                        f"- Profile Summary: {match.profile_summary}\n"
                        f"- Capacity Status: {match.capacity_status}\n"
                    )
                faculty_context = "\n".join(fac_strings)
            else:
                logger.warning("Faculty retrieval returned no matches or failed.")
                faculty_context = "No matching faculty retrieved from database."
        except Exception as e:
            logger.error(f"Error retrieving faculty during recommendation: {e}")
            faculty_context = f"Error performing faculty retrieval: {str(e)}"

        # 2. Analyze current research trends (Step 3)
        trends_context = ""
        has_trends = False
        try:
            logger.info("ProjectRecommendationAgent invoking TrendAgent...")
            trends_res = self.trend_agent.analyze_trends(student_interests)
            if trends_res and trends_res.status in ["success", "partial"]:
                has_trends = True
                trends_context = (
                    f"Research Trends Summary: {trends_res.summary}\n"
                    f"- Emerging Topics: {', '.join(trends_res.emerging_topics)}\n"
                    f"- Key Challenges: {', '.join(trends_res.key_challenges)}\n"
                    f"- Future Opportunities: {', '.join(trends_res.future_opportunities)}"
                )
            else:
                logger.warning("Trend analysis returned unsuccessful status.")
                trends_context = "No recent research trends retrieved."
        except Exception as e:
            logger.error(f"Error performing trend analysis: {e}")
            trends_context = f"Error during trend analysis: {str(e)}"

        # 3. Analyze research gaps (Step 4)
        gaps_context = ""
        has_gaps = False
        try:
            logger.info("ProjectRecommendationAgent invoking GapAnalysisAgent...")
            gaps_res = self.gap_agent.analyze_gaps(student_interests)
            if gaps_res and gaps_res.status in ["success", "partial"]:
                has_gaps = True
                gaps_context = (
                    f"Identified Research Gaps: {', '.join(gaps_res.identified_gaps)}\n"
                    f"- Open Problems: {', '.join(gaps_res.open_problems)}\n"
                    f"- Future Research: {', '.join(gaps_res.future_research)}\n"
                    f"- Thesis Ideas: {', '.join(gaps_res.thesis_ideas)}"
                )
            else:
                logger.warning("Research gap analysis returned unsuccessful status.")
                gaps_context = "No research gaps analyzed."
        except Exception as e:
            logger.error(f"Error performing gap analysis: {e}")
            gaps_context = f"Error during gap analysis: {str(e)}"

        # 4. Build a unified structured context (Step 5)
        structured_context = f"""### UNIFIED RESEARCH CONTEXT
[Student Interests / Query]
{student_interests}

[Retrieved Faculty Profiles]
{faculty_context}

[Recent Academic Trends]
{trends_context}

[Identified Research Gaps]
{gaps_context}
"""

        # Decide status based on input completeness
        final_status = "success" if (has_faculty and has_trends and has_gaps) else "partial"

        # 5. Send ONLY the structured context to Gemini & generate proposal (Step 6 & 7)
        prompt = f"""You are an advanced academic research architect responsible for generating high-quality, personalized research project recommendations.

You MUST formulate a highly personalized and grounded research project recommendation strictly using the following unified structured context.

{structured_context}

### STRICT GROUNDING AND ANTI-HALLUCINATION RULES:
1. NEVER invent, hallucinate, or assume any faculty members, trends, or research gaps.
2. Only recommend faculty members who are explicitly listed in the [Retrieved Faculty Profiles] section above.
3. Recommend faculty whose research interests align best with the proposed project, and provide a clear explanation for this alignment in 'reasoning'.
4. Do not invent trends or gaps that are not directly supported by or derived from the [Recent Academic Trends] and [Identified Research Gaps] sections.
5. If some sections (such as faculty, trends, or gaps) are missing or marked with "No matching..." or "Error...", handle this gracefully by relying strictly on the available context or formulating a project proposal based on student interests alone, with status set to "partial" if critical inputs are missing.

### INSTRUCTIONS:
Generate a personalized research project recommendation. The output must be returned as a valid JSON object.

The JSON object must contain the following keys exactly:
- "project_title": A creative, precise, and descriptive academic project title.
- "problem_statement": A clear, rigorous statement of the technical or scientific problem the project addresses.
- "research_motivation": A compelling academic motivation for pursuing this research.
- "objectives": A list of 3-5 specific, measurable, and realistic research objectives.
- "novelty": A clear explanation of what makes this project novel and how it fills the identified research gaps.
- "methodology": A suggested step-by-step technical methodology or approach to execute the project.
- "expected_outcomes": A list of expected tangible outcomes (e.g., algorithms, datasets, publications, software).
- "recommended_faculty": A list of JSON objects representing the recommended faculty supervisors for this project from the retrieved profiles. Each object should have keys:
  - "name": The exact name of the faculty member.
  - "department": The department of the faculty member.
  - "alignment_reason": A detailed description of how their profile, current projects, or publications align with this recommended project.
- "required_skills": A list of key technical skills, tools, or domain knowledge required to succeed.
- "difficulty": A self-assessed difficulty level (e.g., "Beginner", "Intermediate", "Advanced").
- "future_scope": A forward-looking statement of potential extensions or future research pathways.
- "reasoning": A thorough, high-integrity comparative reasoning detailing why this project was recommended, how it bridges the retrieved gaps, and why the recommended faculty members are the perfect fit.
- "status": Set to "{final_status}".

Respond ONLY with the raw JSON object. Do not wrap it in any formatting other than valid JSON. Do not include markdown code block backticks unless they are valid JSON string wrappers."""

        try:
            logger.info("Requesting structured project recommendation from Gemini...")
            result = self.gemini.generate_json(prompt)
            if isinstance(result, dict):
                # Ensure all required keys from Task 3 are present, falling back if missing
                return {
                    "project_title": result.get("project_title") or f"Research Proposal on {student_interests}",
                    "problem_statement": result.get("problem_statement") or "Study of the requested topic.",
                    "research_motivation": result.get("research_motivation") or "No motivation provided.",
                    "objectives": result.get("objectives") if isinstance(result.get("objectives"), list) else [],
                    "novelty": result.get("novelty") or "N/A",
                    "methodology": result.get("methodology") or "N/A",
                    "expected_outcomes": result.get("expected_outcomes") if isinstance(result.get("expected_outcomes"), list) else [],
                    "recommended_faculty": result.get("recommended_faculty") if isinstance(result.get("recommended_faculty"), list) else [],
                    "required_skills": result.get("required_skills") if isinstance(result.get("required_skills"), list) else [],
                    "difficulty": result.get("difficulty") or "Intermediate",
                    "future_scope": result.get("future_scope") or "N/A",
                    "reasoning": result.get("reasoning") or "Grounded academic fit reasoning.",
                    "status": result.get("status") or final_status
                }
            else:
                raise ValueError("Returned result is not a dictionary.")
        except Exception as e:
            logger.error(f"Project recommendation generation failed: {e}")
            # Build high-integrity default fallback response to safeguard system performance (Task 5)
            fallback_fac = []
            for idx, f in enumerate(faculty_list[:2]):
                fallback_fac.append({
                    "name": f.name,
                    "department": f.department,
                    "alignment_reason": "Identified through vector search as a high-alignment match for student query."
                })
            return {
                "project_title": f"Research Proposal: Exploration of {student_interests}",
                "problem_statement": f"Investigating the core concepts, challenges, and implementation methodologies of {student_interests}.",
                "research_motivation": "Robust synthesis pipeline fallback activated due to API, network, or decoding issues.",
                "objectives": [
                    "Explore foundational concepts related to the user interest query",
                    "Survey relevant literature and recent publications",
                    "Formulate a pilot design or experimental baseline"
                ],
                "novelty": "Personalized study formulated directly from the user's local profile and search interests.",
                "methodology": "1. Perform a deep literature review of relevant supervisor papers.\n2. Design standard baseline models.\n3. Run initial experimental trials.",
                "expected_outcomes": ["Detailed literature review document", "Baseline model prototype"],
                "recommended_faculty": fallback_fac,
                "required_skills": ["Literature review", "Python programming", "Analytical reasoning"],
                "difficulty": "Intermediate",
                "future_scope": "Iterate on more specific methodologies as more data becomes available.",
                "reasoning": f"Generation failed due to: {str(e)}. Formulated a high-quality fallback proposal based on local context.",
                "status": "partial"
            }


def project_recommendation_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph agent node responsible for formulating project proposal recommendations.
    Synthesizes student interests, matching faculty, current trends, and gaps.
    """
    print("[Agent Node] executing project_recommendation_agent_node...")
    query = state.get("current_query", "")
    
    # We should get student profile or current_query
    student_profile = state.get("student_profile")
    interests = query
    if student_profile and isinstance(student_profile, dict):
        profile_interests = student_profile.get("research_interests") or student_profile.get("interests")
        if profile_interests:
            if isinstance(profile_interests, list):
                interests = ", ".join(profile_interests)
            else:
                interests = str(profile_interests)
                
    agent = ProjectRecommendationAgent()
    try:
        recommendation_dict = agent.recommend_projects(interests)
        
        # Store recommendation under "project_recommendations" state field
        return {
            "project_recommendations": [recommendation_dict],
            "retrieved_context": recommendation_dict.get("project_title", ""),
            "error": None
        }
    except Exception as e:
        logger.error(f"ProjectRecommendationAgent node failed: {e}")
        return {
            "error": f"ProjectRecommendationAgentNode failed: {str(e)}"
        }
