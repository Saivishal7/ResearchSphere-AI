# agents/professor_agent.py
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from graph.state import ResearchState
from tools.gemini_service import GeminiService
from agents.trend_agent import TrendAgent
from agents.gap_agent import GapAnalysisAgent
from agents.collaboration_agent import CollaborationAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProfessorAgentResponse(BaseModel):
    """
    Strongly typed structured response model for the ProfessorAgent orchestrator.
    Guarantees reliable serialization and type-safe downstream routing.
    """
    query: str
    intent: str
    analysis: Dict[str, Any]
    recommendations: str
    follow_up_questions: List[str]
    status: str


class ProfessorAgent:
    """
    The Professor Agent: Orchestration-only agent.
    Responsible for classifying academic queries, delegating to specialized services
    (TrendAgent, GapAnalysisAgent, CollaborationAgent), and synthesizing findings.
    
    Adheres strictly to the architectural boundary of no direct retrieval,
    no database access, and no direct search tool interactions.
    """
    def __init__(self):
        self.gemini = GeminiService()
        self.trend_agent = TrendAgent()
        self.gap_analysis_agent = GapAnalysisAgent()
        self.collaboration_agent = CollaborationAgent()

    def _classify_intent(self, query: str) -> str:
        """
        Interrogates GeminiService to classify the query intent into exactly one of:
        research_trends, research_gap, collaboration, general_query.
        Includes high-reliability fallback.
        """
        if not self.gemini.health_check():
            logger.warning("GeminiService health check failed. Using fallback intent classification.")
            return self._fallback_classify_intent(query)

        try:
            prompt = f"""You are an advanced academic intelligence classifier.
Your task is to classify a professor's query into exactly one of the following categories:
- "research_trends" (if they are asking about trending research topics, emerging areas, or future directions)
- "research_gap" (if they are asking about research gaps, underserved problems, or lab research direction)
- "collaboration" (if they are asking about collaboration opportunities, finding collaborators, or who to work with)
- "general_query" (if none of the above fit)

USER QUERY:
"{query}"

Respond ONLY with a valid JSON object matching this schema:
{{
  "intent": "research_trends" | "research_gap" | "collaboration" | "general_query"
}}"""
            logger.info("Classifying professor query intent via Gemini...")
            result = self.gemini.generate_json(prompt)
            
            if isinstance(result, dict) and "intent" in result:
                intent = result["intent"]
                if intent in ["research_trends", "research_gap", "collaboration", "general_query"]:
                    logger.info(f"Classified intent: '{intent}'")
                    return intent
            
            return self._fallback_classify_intent(query)
        except Exception as e:
            logger.warning(f"Error classifying intent with Gemini: {e}. Falling back.")
            return self._fallback_classify_intent(query)

    def _fallback_classify_intent(self, query: str) -> str:
        """
        Rule-based backup classification system based on key phrase heuristics.
        """
        q_lower = query.lower()
        if any(w in q_lower for w in ["trend", "emerging", "emerge", "hot topic", "direction", "pursue", "popular"]):
            return "research_trends"
        elif any(w in q_lower for w in ["gap", "underserved", "opportunity", "unexplored", "limitations", "weakness"]):
            return "research_gap"
        elif any(w in q_lower for w in ["collaborate", "collaboration", "partner", "co-author", "who should i work with", "colleague"]):
            return "collaboration"
        return "general_query"

    def process_professor_request(self, query: str) -> ProfessorAgentResponse:
        """
        Executes the master orchestration flow:
        1. Classify intent.
        2. Route to specialized service placeholder.
        3. Retrieve structured findings.
        4. Synthesize findings using Gemini.
        5. Structure and return response.
        """
        if not query or not query.strip():
            return ProfessorAgentResponse(
                query="",
                intent="general_query",
                analysis={"error": "Empty query provided"},
                recommendations="Please enter a valid academic research inquiry.",
                follow_up_questions=["What research domain would you like to explore?"],
                status="failed"
            )

        # 1. Classify Intent
        intent = self._classify_intent(query)

        # 2. Route & Receive Structured Output
        try:
            if intent == "research_trends":
                agent_res = self.trend_agent.analyze_trends(query)
                analysis_output = agent_res.model_dump() if hasattr(agent_res, "model_dump") else agent_res.dict()
            elif intent == "research_gap":
                agent_res = self.gap_analysis_agent.analyze_gaps(query)
                analysis_output = agent_res.model_dump() if hasattr(agent_res, "model_dump") else agent_res.dict()
            elif intent == "collaboration":
                agent_res = self.collaboration_agent.suggest_collaborations(query)
                analysis_output = agent_res.model_dump() if hasattr(agent_res, "model_dump") else agent_res.dict()
            else:
                analysis_output = {
                    "status": "general",
                    "message": "General academic query received. No specialized analysis sub-module triggered.",
                    "data": {"query": query}
                }
        except Exception as e:
            logger.error(f"Error executing specialist service: {e}")
            analysis_output = {
                "status": "error",
                "message": f"Specialist agent failure: {str(e)}",
                "data": {}
            }

        # 3. Request Gemini to generate a concise, professional summary
        recommendations = ""
        if self.gemini.health_check():
            try:
                # Send ONLY the retrieved structured info to generate_response
                prompt_summary = f"""You are a professional research advisory assistant.
Here is the retrieved structured analysis data:
{json.dumps(analysis_output, indent=2)}

Please generate a concise, professional, and highly strategic summary of recommendations for the professor based strictly on this data. 
Do not assume, extrapolate, or introduce outside facts. If the report contains placeholder markers, acknowledge that these findings are initial configurations that will be deepened in the future.
"""
                logger.info("Requesting professional summary from Gemini...")
                recommendations = self.gemini.generate_response(prompt_summary)
            except Exception as e:
                logger.error(f"Failed to generate summary with Gemini: {e}")
                recommendations = f"Specialist results successfully loaded. Recommendations generation encountered an exception: {e}"
        else:
            recommendations = "Specialist results loaded. Gemini service is currently offline or unconfigured to generate natural language summaries."

        # 4. Formulate follow-up questions statically based on intent for maximum precision
        if intent == "research_trends":
            follow_up_questions = [
                "Would you like to analyze trend publications in a specific venue like NeurIPS?",
                "Shall we cross-reference these trends with your current lab projects?",
                "Would you like to search Semantic Scholar for the latest preprints on these topics?"
            ]
        elif intent == "research_gap":
            follow_up_questions = [
                "Would you like to check if any of your department colleagues are working on these gaps?",
                "Shall we draft a project proposal framework for one of these underserved areas?",
                "Do you want to retrieve recent literature addressing these specific limitations?"
            ]
        elif intent == "collaboration":
            follow_up_questions = [
                "Would you like to view the full profile and ongoing projects of these potential collaborators?",
                "Shall we draft an introductory outreach email to one of these professors?",
                "Do you want to see their co-authorship history?"
            ]
        else:
            follow_up_questions = [
                "How can I help you optimize your research operations today?",
                "Would you like to run a trend analysis or look for collaboration opportunities?"
            ]

        status = "success" if "error" not in analysis_output else "partial"

        return ProfessorAgentResponse(
            query=query,
            intent=intent,
            analysis=analysis_output,
            recommendations=recommendations,
            follow_up_questions=follow_up_questions,
            status=status
        )


def professor_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph agent node responsible for analyzing professor requests.
    Orchestrates the ProfessorAgent to classify, delegate, and summarize research queries.
    """
    print("[Agent Node] executing professor_agent_node...")
    query = state.get("current_query", "")
    
    agent = ProfessorAgent()
    try:
        response = agent.process_professor_request(query)
        
        updates: Dict[str, Any] = {
            "intent": response.intent,
            "retrieved_context": response.recommendations,
            "error": None
        }

        # Update specific state keys based on intent
        if response.intent == "research_trends":
            updates["research_trends"] = response.analysis
        elif response.intent == "research_gap":
            updates["research_gaps"] = response.analysis
        elif response.intent == "collaboration":
            # Extract list from the placeholder data schema or real schema
            if "recommended_collaborators" in response.analysis:
                updates["collaboration_suggestions"] = response.analysis["recommended_collaborators"]
            else:
                data = response.analysis.get("data", {})
                suggestions = data.get("suggested_collaborators", [])
                updates["collaboration_suggestions"] = suggestions

        return updates
    except Exception as e:
        logger.error(f"Professor Agent node execution error: {e}")
        return {
            "error": f"ProfessorAgentNode failed: {str(e)}"
        }
