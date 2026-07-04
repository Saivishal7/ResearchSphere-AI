# agents/collaboration_agent.py
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from graph.state import ResearchState
from tools.gemini_service import GeminiService
from ingestion.load_faculty import load_all_faculty_profiles, FacultyProfile

# Configure logging
logger = logging.getLogger("CollaborationAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CollaboratorMatch(BaseModel):
    """
    Sub-model representing a recommended collaborator and their research alignment details.
    """
    name: str = Field(..., description="The name of the potential collaborator.")
    department: str = Field(..., description="The department of the potential collaborator.")
    designation: str = Field(..., description="Their academic designation (e.g., Professor, Associate Professor).")
    specialty: str = Field(..., description="The primary technical or research specialty of the candidate.")
    complementary_strength: str = Field(..., description="Explanation of how their expertise complements the query professor.")
    alignment_score: float = Field(..., description="Quantified match alignment score (0.0 to 1.0) computed by the model.")
    joint_projects: List[str] = Field(default_factory=list, description="A list of proposed joint projects or research directions.")


class CollaborationAgentResponse(BaseModel):
    """
    Strongly typed, structured output model representing collaboration recommendations.
    Provides complete type safety and structural validation.
    """
    query: str = Field(..., description="The original query or target research interests.")
    recommended_collaborators: List[CollaboratorMatch] = Field(
        default_factory=list,
        description="Ranked list of recommended collaborators with details and joint project proposals."
    )
    shared_interests: List[str] = Field(default_factory=list, description="Identified common research topics or keywords.")
    complementary_expertise: List[str] = Field(default_factory=list, description="Areas of complementary or interdisciplinary expertise.")
    possible_joint_projects: List[str] = Field(default_factory=list, description="High-level suggested collaborative project proposals.")
    reasoning: str = Field(..., description="Comprehensive justification of rankings, selection methodology, and opportunities.")
    status: str = Field(..., description="Status indicator representing execution status ('success', 'partial', 'no_results', 'failed').")


class CollaborationAgent:
    """
    The Collaboration Recommendation Agent.
    Responsible for identifying potential faculty collaborations based on complementary expertise.
    
    Enforces absolute architectural boundaries:
    - Never performs web search or calls Tavily.
    - Never accesses ChromaDB directly.
    - Uses only the local faculty dataset.
    - Employs deterministic Python heuristics for candidate pre-ranking.
    - Uses GeminiService ONLY to reason about pre-filtered candidates.
    """
    def __init__(self):
        self.gemini = GeminiService()

    def _compute_python_heuristic_score(self, target_text: str, profile: FacultyProfile) -> float:
        """
        Calculates a programmatic overlap match score between target research text and a faculty profile.
        Used to filter and pre-rank candidate profiles to optimize the LLM context size.
        """
        target_words = set(re.findall(r'\w+', target_text.lower()))
        if not target_words:
            return 0.0

        interests_text = " ".join(profile.research_interests).lower()
        keywords_text = " ".join(profile.keywords).lower()
        summary_text = profile.profile_summary.lower()
        projects_text = " ".join([p.title + " " + p.description for p in profile.current_projects]).lower()
        pubs_text = " ".join([pub.title + " " + pub.conference_or_journal for pub in profile.publications]).lower()

        score = 0.0
        for word in target_words:
            if len(word) < 3:  # Skip trivial stopwords/short tokens
                continue
            if word in interests_text:
                score += 3.0
            if word in keywords_text:
                score += 2.0
            if word in summary_text:
                score += 1.5
            if word in projects_text:
                score += 1.0
            if word in pubs_text:
                score += 0.5

        return score

    def suggest_collaborations(self, query: str) -> CollaborationAgentResponse:
        """
        Orchestrates the collaborator recommendation workflow.
        """
        if not query or not query.strip():
            logger.warning("Collaboration recommendation invoked with empty query.")
            return CollaborationAgentResponse(
                query="",
                recommended_collaborators=[],
                shared_interests=[],
                complementary_expertise=[],
                possible_joint_projects=[],
                reasoning="Empty query provided. Please provide research interests or a faculty name to suggest collaborations.",
                status="failed"
            )

        # 1. Load the complete local faculty dataset
        try:
            profiles = load_all_faculty_profiles("data/faculty")
        except Exception as e:
            logger.error(f"Failed to load local faculty profiles: {e}")
            return CollaborationAgentResponse(
                query=query,
                recommended_collaborators=[],
                shared_interests=[],
                complementary_expertise=[],
                possible_joint_projects=[],
                reasoning=f"Failed to load local faculty dataset. Internal error: {str(e)}",
                status="failed"
            )

        if not profiles:
            logger.warning("No profiles found in the local faculty directory.")
            return CollaborationAgentResponse(
                query=query,
                recommended_collaborators=[],
                shared_interests=[],
                complementary_expertise=[],
                possible_joint_projects=[],
                reasoning="No faculty profiles exist in the system dataset.",
                status="no_results"
            )

        # 2. Identify if the query is referring to an existing professor in the dataset
        source_profile: Optional[FacultyProfile] = None
        query_lower = query.lower()
        for p in profiles:
            name_lower = p.name.lower()
            # Strict or fuzzy match against complete name or last name
            if name_lower in query_lower or (len(p.name.split()) > 1 and p.name.split()[-1].lower() in query_lower):
                source_profile = p
                logger.info(f"Resolved query to existing faculty member: Dr. {p.name}")
                break

        # 3. Determine the target research interest payload & list of potential candidates
        if source_profile:
            # Recommending collaborations for a specific registered professor
            projects_desc = " ".join([f"{proj.title} {proj.description}" for proj in source_profile.current_projects])
            pubs_desc = " ".join([f"{pub.title} {pub.conference_or_journal}" for pub in source_profile.publications])
            target_interests = f"{' '.join(source_profile.research_interests)} {' '.join(source_profile.keywords)} {source_profile.profile_summary} {projects_desc} {pubs_desc}"
            
            # Exclude the source professor from being their own collaborator
            candidates = [p for p in profiles if p.faculty_id != source_profile.faculty_id]
        else:
            # Query is generic research interests/topics
            target_interests = query
            candidates = profiles

        # 4. Programmatic pre-ranking and selection using Python heuristics
        ranked_candidates = []
        for p in candidates:
            score = self._compute_python_heuristic_score(target_interests, p)
            ranked_candidates.append((score, p))

        # Sort descending by match score
        ranked_candidates.sort(key=lambda x: x[0], reverse=True)

        # Slice the top 6 candidates to provide a high-relevance pool for LLM synthesis
        top_candidates = [p for _, p in ranked_candidates[:6]]

        # 5. Build structured context representation for the candidate pool
        context_blocks = []
        for p in top_candidates:
            proj_list = [f"  - Project: {proj.title}\n    Description: {proj.description[:120]}..." for proj in p.current_projects[:2]]
            pub_list = [f"  - Publication: {pub.title} ({pub.year}) inside {pub.conference_or_journal}" for pub in p.publications[:2]]
            
            proj_str = "\n".join(proj_list) if proj_list else "  - None documented"
            pub_str = "\n".join(pub_list) if pub_list else "  - None documented"
            
            p_block = f"""Faculty ID: {p.faculty_id}
Name: {p.name}
Department: {p.department}
Designation: {p.designation}
Research Interests: {", ".join(p.research_interests)}
Keywords: {", ".join(p.keywords)}
Profile Summary: {p.profile_summary}
Recent Projects:
{proj_str}
Recent Publications:
{pub_str}
---"""
            context_blocks.append(p_block)

        candidates_context = "\n".join(context_blocks)

        if source_profile:
            query_details = f"""SOURCE PROFESSOR:
Name: {source_profile.name}
Department: {source_profile.department}
Designation: {source_profile.designation}
Research Interests: {", ".join(source_profile.research_interests)}
Keywords: {", ".join(source_profile.keywords)}
Profile Summary: {source_profile.profile_summary}"""
        else:
            query_details = f"TARGET RESEARCH TOPIC OR INTERESTS: {query}"

        # 6. Formulate high-integrity prompt for Gemini Service
        prompt = f"""You are an elite academic network architect and institutional advisor.
Evaluate and suggest potential collaborations from the provided faculty candidates pool based ONLY on complementary research expertise, overlapping interests, or interdisciplinary opportunities.

EVALUATION PROFILE DETAILS:
{query_details}

CANDIDATES POOL CONTEXT:
{candidates_context}

STRICT OPERATIONAL DIRECTIVES:
1. Base your recommendations, alignments, and joint research projects ONLY on the provided Candidates Pool Context and Evaluation Profile. Do NOT assume, extrapolate, or hallucinate credentials or publications.
2. Rank the candidates starting with rank 1 as the highest suitability. Recommend only up to 4 best fits.
3. If no candidate represents a suitable match, return an empty recommended list and provide an objective reasoning statement.
4. Each recommended collaborator must include a calculated alignment_score (0.0 to 1.0) based on their actual overlap and complementary strengths, a clear explanation of their complementary strength, their key specialty, and concrete proposed joint projects.

Respond strictly with a valid JSON object matching the following schema. Do not include markdown backticks or wrappers:
{{
  "query": "string (the original query)",
  "recommended_collaborators": [
    {{
      "name": "string (collaborator name)",
      "department": "string",
      "designation": "string",
      "specialty": "string (e.g. Edge Hardware Accelerators)",
      "complementary_strength": "string (why and how their expertise complements the source profile based ONLY on context)",
      "alignment_score": float,
      "joint_projects": [
        "string (concrete collaborative project proposal or research direction)"
      ]
    }}
  ],
  "shared_interests": [
    "string (common research topics found between profiles)"
  ],
  "complementary_expertise": [
    "string (specific technical areas where skills complement each other)"
  ],
  "possible_joint_projects": [
    "string (high-level joint research project ideas)"
  ],
  "reasoning": "string (comprehensive overview of how the rankings were decided and why they represent high potential)"
}}"""

        # 7. Query Gemini Service for synthesis
        if self.gemini.health_check():
            try:
                logger.info("Invoking Gemini to rank and synthesize collaborator recommendations...")
                res_json = self.gemini.generate_json(prompt)
                
                if not isinstance(res_json, dict):
                    raise ValueError("Gemini response was not a structured dictionary.")

                query_val = res_json.get("query", query)
                raw_collaborators = res_json.get("recommended_collaborators", [])
                shared = res_json.get("shared_interests", [])
                complementary = res_json.get("complementary_expertise", [])
                joint_projects = res_json.get("possible_joint_projects", [])
                reasoning_val = res_json.get("reasoning", "Successfully calculated academic alignments.")

                # Ensure perfect type validation and deserialization
                collaborators = []
                for rc in raw_collaborators:
                    try:
                        collaborators.append(
                            CollaboratorMatch(
                                name=rc.get("name", "Unknown Professor"),
                                department=rc.get("department", "Unknown Department"),
                                designation=rc.get("designation", "Faculty"),
                                specialty=rc.get("specialty", "Unspecified"),
                                complementary_strength=rc.get("complementary_strength", "Complementary skills found."),
                                alignment_score=float(rc.get("alignment_score", 0.5)),
                                joint_projects=rc.get("joint_projects", [])
                            )
                        )
                    except Exception as parse_err:
                        logger.warning(f"Skipped malformed collaborator match: {parse_err}")

                return CollaborationAgentResponse(
                    query=query_val,
                    recommended_collaborators=collaborators,
                    shared_interests=shared if isinstance(shared, list) else [],
                    complementary_expertise=complementary if isinstance(complementary, list) else [],
                    possible_joint_projects=joint_projects if isinstance(joint_projects, list) else [],
                    reasoning=reasoning_val,
                    status="success" if collaborators else "no_results"
                )

            except Exception as e:
                logger.error(f"Fuzzy LLM collaboration generation failed: {e}. Executing clean fallback...")
        else:
            logger.warning("GeminiService health check failed. Executing clean static fallback...")

        # 8. Statically defined robust fallback mapping
        fallback_matches = []
        for score, p in ranked_candidates[:2]:
            if score > 0.0:
                fallback_matches.append(
                    CollaboratorMatch(
                        name=p.name,
                        department=p.department,
                        designation=p.designation,
                        specialty=p.research_interests[0] if p.research_interests else "Research Specialist",
                        complementary_strength=f"Exhibits keyword and topic alignments with '{topic_val if 'topic_val' in locals() else query}' through matching publications and projects.",
                        alignment_score=round(min(0.95, 0.4 + (score * 0.05)), 2),
                        joint_projects=[f"Joint exploratory research combining {query[:30]} and {p.research_interests[0] if p.research_interests else 'related topics'}"]
                    )
                )

        return CollaborationAgentResponse(
            query=query,
            recommended_collaborators=fallback_matches,
            shared_interests=[query] if not source_profile else source_profile.research_interests[:2],
            complementary_expertise=[p.research_interests[0] for _, p in ranked_candidates[:2] if p.research_interests],
            possible_joint_projects=[f"Collaborative initiative on {query}"],
            reasoning=f"Faculty database loaded and queried. Standard fallback recommendations generated heuristically without full LLM synthesis due to unconfigured services or schema parsing exceptions.",
            status="partial" if fallback_matches else "no_results"
        )


def collaboration_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph agent node responsible for discovering and recommending collaborators.
    Orchestrates the CollaborationAgent to evaluate and output suggestions to system state.
    """
    print("[Agent Node] executing collaboration_agent_node...")
    query = state.get("current_query", "")
    
    agent = CollaborationAgent()
    try:
        response = agent.suggest_collaborations(query)
        # Convert Pydantic models to dicts for LangGraph state storage compatibility
        serialized_collaborators = []
        for rc in response.recommended_collaborators:
            serialized_collaborators.append({
                "name": rc.name,
                "department": rc.department,
                "designation": rc.designation,
                "specialty": rc.specialty,
                "complementary_strength": rc.complementary_strength,
                "alignment_score": rc.alignment_score,
                "joint_projects": rc.joint_projects
            })

        return {
            "collaboration_suggestions": serialized_collaborators,
            "retrieved_context": response.reasoning,
            "error": None if response.status in ["success", "partial"] else f"Collaboration discovery failed with status: {response.status}"
        }
    except Exception as e:
        logger.error(f"Collaboration Agent node execution error: {e}")
        return {
            "error": f"CollaborationAgentNode failed: {str(e)}"
        }
