# agents/trend_agent.py
import logging
import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from config import TAVILY_API_KEY
from tools.gemini_service import GeminiService
from graph.state import ResearchState

# Configure logging
logger = logging.getLogger("TrendAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TrendAgentResponse(BaseModel):
    """
    Strongly typed, structured output model representing modern research trend analysis.
    This acts as a reliable payload containing summary and future direction highlights.
    """
    topic: str = Field(..., description="The query topic of the trend analysis.")
    summary: str = Field(..., description="A professional concise summary of the trend based strictly on search context.")
    emerging_topics: List[str] = Field(default_factory=list, description="A list of newly emerging sub-fields.")
    key_challenges: List[str] = Field(default_factory=list, description="Core challenges or barriers identified.")
    future_opportunities: List[str] = Field(default_factory=list, description="Key potential research opportunities.")
    references: List[Dict[str, str]] = Field(default_factory=list, description="Context sources with titles and URLs.")
    status: str = Field(..., description="Status indicator representing execution success ('success', 'partial', 'no_results', 'failed').")


class TrendAgent:
    """
    The Trend Analysis Agent: responsible for querying recent literature/web indices via Tavily Search,
    compiling robust research contexts, and synthesizing emerging academic topics via GeminiService.
    
    Enforces a strict RAG boundary: Gemini only summarizes the explicitly retrieved search context
    to minimize hallucinations.
    """
    def __init__(self):
        self.gemini = GeminiService()
        self.tavily_api_key = TAVILY_API_KEY

    def _perform_tavily_search(self, topic: str) -> List[Dict[str, Any]]:
        """
        Queries the Tavily Search API. Returns a list of result dictionaries.
        Supports automatic fallback to direct HTTP requests in case the SDK is missing/unconfigured.
        """
        if not self.tavily_api_key:
            logger.warning("Tavily API key is missing. Skipping external search query.")
            return []

        try:
            # 1. Attempt using the official Tavily SDK
            try:
                from tavily import TavilyClient
                logger.info(f"Using Tavily SDK to search for topic: '{topic}'")
                client = TavilyClient(api_key=self.tavily_api_key)
                response = client.search(query=topic, max_results=5, search_depth="advanced")
                results = response.get("results", [])
                if results:
                    return results
            except ImportError:
                logger.info("Tavily Python SDK not available. Using direct HTTP REST requests.")

            # 2. Direct HTTP requests fallback
            url = "https://api.tavily.com/search"
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": self.tavily_api_key,
                "query": topic,
                "max_results": 5,
                "search_depth": "advanced"
            }
            logger.info(f"Executing direct POST request to Tavily for topic: '{topic}'")
            res = requests.post(url, json=payload, headers=headers, timeout=12)
            if res.status_code == 200:
                data = res.json()
                return data.get("results", [])
            else:
                logger.error(f"Tavily REST endpoint failed with status code {res.status_code}: {res.text}")
                return []
        except Exception as e:
            logger.error(f"Error during Tavily search execution: {e}")
            return []

    def analyze_trends(self, topic: str) -> TrendAgentResponse:
        """
        Orchestrates the entire trend analysis workflow.
        """
        # Handle empty query gracefully
        if not topic or not topic.strip():
            logger.warning("Trend analysis invoked with empty topic query.")
            return TrendAgentResponse(
                topic="",
                summary="The requested topic is empty. Please provide a valid research topic to analyze.",
                emerging_topics=[],
                key_challenges=[],
                future_opportunities=[],
                references=[],
                status="failed"
            )

        # 1. Execute Tavily search to fetch latest context
        search_results = self._perform_tavily_search(topic)

        # 2. Handle no search results / offline mode gracefully
        if not search_results:
            logger.warning(f"No web results retrieved from Tavily for topic: '{topic}'")
            return TrendAgentResponse(
                topic=topic,
                summary=f"Unable to retrieve search trends for '{topic}'. The external search service is currently offline or returned no results. Please ensure a valid Tavily API key is configured.",
                emerging_topics=[],
                key_challenges=["External search service offline or rate-limited"],
                future_opportunities=[],
                references=[],
                status="no_results"
            )

        # 3. Build context and references
        context_items = []
        references = []
        for r in search_results:
            title = r.get("title", "Untitled Web Source")
            url = r.get("url", "")
            content = r.get("content", "")
            if content:
                context_items.append(f"Title: {title}\nURL: {url}\nContent: {content}\n---")
                references.append({"title": title, "url": url})

        retrieved_context = "\n".join(context_items)

        # 4. Formulate prompt for Gemini with zero-hallucination guardrails
        prompt = f"""You are an elite academic researcher and technical analyst.
Generate a structured research trend report for the specified topic based ONLY on the provided search results context below.

TOPIC TO ANALYZE:
"{topic}"

SEARCH RESULTS CONTEXT:
{retrieved_context}

STRICT OPERATIONAL DIRECTIVES:
1. Base your summaries and key indicators ONLY on the retrieved search results. Do NOT assume, speculate, or introduce external facts.
2. If any field (emerging directions, challenges, opportunities) cannot be answered with high confidence using only the context, clearly note that the information is unavailable in the retrieved results.
3. Be objective, concise, and academically rigorous.

Respond strictly with a valid JSON object matching this schema (Do not include markdown wrappers, backticks, or extraneous content):
{{
  "topic": "string",
  "summary": "string (comprehensive, technical overview of current trends based ONLY on context)",
  "emerging_topics": [
    "string (sub-topic or emerging direction found in context)"
  ],
  "key_challenges": [
    "string (unsolved challenge or friction point found in context)"
  ],
  "future_opportunities": [
    "string (concrete research opportunity found in context)"
  ]
}}"""

        # 5. Invoke Gemini Service to synthesize trend metrics
        try:
            logger.info("Sending retrieved search context to GeminiService for trend synthesis...")
            res_json = self.gemini.generate_json(prompt)
            
            if not isinstance(res_json, dict):
                raise ValueError("Synthesis payload from Gemini was not structured JSON.")

            topic_val = res_json.get("topic", topic)
            summary_val = res_json.get("summary", "Trend analysis compilation finished.")
            emerging = res_json.get("emerging_topics", [])
            challenges = res_json.get("key_challenges", [])
            opportunities = res_json.get("future_opportunities", [])

            return TrendAgentResponse(
                topic=topic_val,
                summary=summary_val,
                emerging_topics=emerging if isinstance(emerging, list) else [],
                key_challenges=challenges if isinstance(challenges, list) else [],
                future_opportunities=opportunities if isinstance(opportunities, list) else [],
                references=references,
                status="success"
            )
        except Exception as e:
            logger.error(f"Failed to compile trend report via Gemini: {e}. Falling back to default.")
            # Build high-integrity default fallback response to safeguard system performance
            return TrendAgentResponse(
                topic=topic,
                summary=f"Search retrieved latest context successfully, but synthesis failed to generate structured report. Error: {str(e)}",
                emerging_topics=["Examine retrieved search references below."],
                key_challenges=["LLM synthesis exception"],
                future_opportunities=[],
                references=references,
                status="partial"
            )


def trend_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph agent node responsible for analyzing research trends.
    Uses TrendAgent to query Tavily and generate clean summaries.
    """
    print("[Agent Node] executing trend_agent_node...")
    query = state.get("current_query", "")
    
    agent = TrendAgent()
    try:
        response = agent.analyze_trends(query)
        # Convert response to dictionary for state storage
        serialized_trends = response.model_dump()
        return {
            "research_trends": serialized_trends,
            "retrieved_context": response.summary,
            "error": None if response.status in ["success", "partial"] else f"Trend Analysis failed with status: {response.status}"
        }
    except Exception as e:
        logger.error(f"Trend Agent node execution error: {e}")
        return {
            "error": f"TrendAgentNode failed: {str(e)}"
        }
