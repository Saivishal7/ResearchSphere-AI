# agents/gap_agent.py
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from tools.gemini_service import GeminiService
from agents.trend_agent import TrendAgent
from graph.state import ResearchState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GapAnalysisAgentResponse(BaseModel):
    """
    Structured response model representing the finalized research gap analysis.
    Provides complete type safety and structural validation of research gap properties.
    """
    topic: str = Field(..., description="The query topic of the research gap analysis.")
    identified_gaps: List[str] = Field(default_factory=list, description="A list of newly identified research gaps derived from current trends.")
    open_problems: List[str] = Field(default_factory=list, description="Core open technical or scientific problems.")
    future_research: List[str] = Field(default_factory=list, description="Concrete future research directions.")
    thesis_ideas: List[str] = Field(default_factory=list, description="Innovative research or thesis topic suggestions.")
    confidence: float = Field(..., description="Self-assessed confidence score based on available data quality (0.0 to 1.0).")
    status: str = Field(..., description="Status indicator of the analysis ('success', 'failed', 'partial').")


class GapAnalysisAgent:
    """
    The Research Gap Analysis Agent.
    Responsible for analyzing retrieved trend information and identifying meaningful research gaps.
    
    Following strict architectural rules:
    - Never performs web search.
    - Never calls Tavily directly.
    - Never retrieves faculty.
    - Never accesses ChromaDB.
    - Delegates trends/retrieval to TrendAgent.
    - Uses Gemini for reasoning and synthesis.
    """
    def __init__(self):
        self.gemini = GeminiService()
        self.trend_agent = TrendAgent()

    def analyze_gaps(self, topic: str) -> GapAnalysisAgentResponse:
        """
        Executes the gap analysis workflow:
        1. Receive a research topic.
        2. Call TrendAgent to discover current trends.
        3. Parse structured trend results.
        4. Build clean trend context.
        5. Query Gemini Service with ONLY retrieved context to discover research gaps.
        """
        if not topic or not topic.strip():
            logger.warning("Gap analysis invoked with empty topic query.")
            return GapAnalysisAgentResponse(
                topic="",
                identified_gaps=[],
                open_problems=[],
                future_research=[],
                thesis_ideas=[],
                confidence=0.0,
                status="failed"
            )

        logger.info(f"Starting gap analysis for topic '{topic}'")

        # 1. Retrieve current trends using TrendAgent
        try:
            logger.info(f"Delegating trend discovery for topic '{topic}' to TrendAgent...")
            trend_results = self.trend_agent.analyze_trends(topic)
            logger.info(f"TrendAgent returned status '{getattr(trend_results, 'status', 'unknown')}' for topic '{topic}'")

        except Exception as e:
            logger.error(f"TrendAgent failed during delegation: {e}")
            return GapAnalysisAgentResponse(
                topic=topic,
                identified_gaps=[f"Trend discovery failed: {e}"],
                open_problems=[],
                future_research=[],
                thesis_ideas=[],
                confidence=0.0,
                status="failed"
            )

        # 2. Check if trend results are valid
        if not trend_results or trend_results.status in ["failed", "no_results"]:
            logger.warning(f"TrendAgent returned invalid or empty results with status '{getattr(trend_results, 'status', 'None')}'.")
            return GapAnalysisAgentResponse(
                topic=topic,
                identified_gaps=[f"No usable trend context was found (status: {getattr(trend_results, 'status', 'unknown')})."],
                open_problems=[],
                future_research=[],
                thesis_ideas=[],
                confidence=0.0,
                status="failed"
            )

        # 3. Build a clean, structured context from trend results
        trend_summary = trend_results.summary
        emerging_topics_str = ", ".join(trend_results.emerging_topics) if trend_results.emerging_topics else "None mentioned"
        key_challenges_str = ", ".join(trend_results.key_challenges) if trend_results.key_challenges else "None mentioned"
        future_opps_str = ", ".join(trend_results.future_opportunities) if trend_results.future_opportunities else "None mentioned"

        trend_context = f"""Research Topic: {trend_results.topic}
Current Trend Summary: {trend_summary}
Emerging Trends/Topics: {emerging_topics_str}
Key Challenges: {key_challenges_str}
Future Opportunities: {future_opps_str}"""

        # 4. Ask Gemini to identify gaps using ONLY that context (preventing hallucinations)
        prompt = f"""You are an elite academic advisor and research strategist.
Your task is to analyze the provided research trend context and identify meaningful research gaps, open problems, future research, and thesis ideas.

STRICT OPERATIONAL DIRECTIVES:
1. Base your analysis and suggestions ONLY on the provided research trend context below. Do NOT assume, extrapolate, or introduce external facts.
2. If the context is insufficient to confidently identify gaps, open problems, or thesis ideas, explicitly state that in your response fields or return empty lists rather than inventing details.
3. Assess the data quality and assign a numerical confidence score between 0.0 (no reliable data) and 1.0 (highly comprehensive data) based ONLY on the depth of the provided context.

RESEARCH TREND CONTEXT:
{trend_context}

Respond strictly with a valid JSON object matching the following schema. Do not include markdown blocks, backticks, or other wrappers:
{{
  "topic": "string",
  "identified_gaps": [
    "string (specific, concrete gap in research derived ONLY from context)"
  ],
  "open_problems": [
    "string (specific open question or challenge mentioned or directly implied in context)"
  ],
  "future_research": [
    "string (promising future research avenue from context)"
  ],
  "thesis_ideas": [
    "string (well-formulated, innovative research or thesis proposal idea based on the gaps)"
  ],
  "confidence": float (between 0.0 and 1.0)
}}"""

        try:
            logger.info("Sending trend context to GeminiService for gap analysis synthesis...")
            res_json = self.gemini.generate_json(prompt)
            logger.info(f"Gemini returned a structured payload for topic '{topic}'")

            if not isinstance(res_json, dict):
                raise ValueError("Response from Gemini was not a structured dictionary.")

            topic_val = res_json.get("topic", topic)
            gaps = res_json.get("identified_gaps", [])
            problems = res_json.get("open_problems", [])
            future_res = res_json.get("future_research", [])
            thesis = res_json.get("thesis_ideas", [])
            confidence_val = res_json.get("confidence", 0.7)

            # Defensive type checks
            if not isinstance(gaps, list): gaps = []
            if not isinstance(problems, list): problems = []
            if not isinstance(future_res, list): future_res = []
            if not isinstance(thesis, list): thesis = []
            try:
                confidence_val = float(confidence_val)
            except (ValueError, TypeError):
                confidence_val = 0.7

            return GapAnalysisAgentResponse(
                topic=topic_val,
                identified_gaps=gaps,
                open_problems=problems,
                future_research=future_res,
                thesis_ideas=thesis,
                confidence=confidence_val,
                status="success"
            )
        except Exception as e:
            logger.error(f"Structured gap analysis compilation failed via Gemini: {e}")
            return GapAnalysisAgentResponse(
                topic=topic,
                identified_gaps=["Error parsing gap analysis synthesis. See log."],
                open_problems=[],
                future_research=[],
                thesis_ideas=[],
                confidence=0.3,
                status="partial"
            )


def gap_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph agent node responsible for analyzing research gaps.
    Orchestrates the GapAnalysisAgent to identify gaps, open problems, and thesis ideas.
    """
    print("[Agent Node] executing gap_agent_node...")
    query = state.get("current_query", "")
    
    agent = GapAnalysisAgent()
    try:
        response = agent.analyze_gaps(query)
        serialized_gaps = response.model_dump()
        
        # Build a clean synthesis/explanation string to store in retrieved_context
        retrieved_context_summary = (
            f"Identified Gaps: {', '.join(response.identified_gaps[:3])}\n"
            f"Open Problems: {', '.join(response.open_problems[:3])}"
        )
        
        return {
            "research_gaps": serialized_gaps,
            "retrieved_context": response.thesis_ideas[0] if response.thesis_ideas else retrieved_context_summary,
            "error": None if response.status in ["success", "partial"] else f"Gap Analysis failed with status: {response.status}"
        }
    except Exception as e:
        logger.error(f"Gap Analysis node execution error: {e}")
        return {
            "error": f"GapAgentNode failed: {str(e)}"
        }
