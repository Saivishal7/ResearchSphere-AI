# agents/faculty_retrieval_agent.py
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

import chromadb
from config import CHROMA_STORE_DIR
from ingestion.load_faculty import FacultyProfile, Project, Publication, load_faculty_profile
from graph.state import ResearchState

# Configure logging
logger = logging.getLogger("FacultyRetrievalAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class FacultyMatch(BaseModel):
    """
    Represents a single retrieved faculty member matching a search query.
    Contains all original faculty fields along with the calculated similarity score.
    """
    faculty_id: str
    name: str
    department: str
    designation: str
    similarity_score: float
    research_interests: List[str]
    profile_summary: str
    current_projects: List[Project]
    publications: List[Publication]
    keywords: List[str]
    capacity_status: str

    @classmethod
    def from_profile(cls, profile: FacultyProfile, similarity_score: float) -> "FacultyMatch":
        """
        Constructs a FacultyMatch instance directly from a validated FacultyProfile.
        """
        return cls(
            faculty_id=profile.faculty_id,
            name=profile.name,
            department=profile.department,
            designation=profile.designation,
            similarity_score=similarity_score,
            research_interests=profile.research_interests,
            profile_summary=profile.profile_summary,
            current_projects=profile.current_projects,
            publications=profile.publications,
            keywords=profile.keywords,
            capacity_status=profile.capacity_status
        )


class FacultyRetrievalResult(BaseModel):
    """
    Structured outcome of a retrieval request, shielding caller code from direct
    exceptions and database-specific query structures.
    """
    matches: List[FacultyMatch] = Field(default_factory=list)
    success: bool
    error_message: Optional[str] = None


class FacultyRetrievalAgent:
    """
    The Faculty Retrieval Agent: responsible strictly for performing local semantic
    retrieval from ChromaDB. It handles embedding generation and optional metadata
    filtering, returning strongly-typed structured results.
    """
    def __init__(self, collection_name: str = "faculty_profiles", db_dir: str = CHROMA_STORE_DIR):
        self.collection_name = collection_name
        self.db_dir = db_dir
        self.client: Optional[chromadb.PersistentClient] = None
        self._initialize_chroma()

    def _initialize_chroma(self) -> None:
        """
        Safely establishes a connection with the local persistent ChromaDB storage.
        """
        try:
            logger.info(f"Initializing ChromaDB Persistent Client at '{self.db_dir}'...")
            self.client = chromadb.PersistentClient(path=self.db_dir)
        except Exception as e:
            logger.error(f"ChromaDB connection failure: {e}")
            self.client = None

    def _build_where_clause(self, department: Optional[str] = None, capacity_status: Optional[str] = None) -> Optional[dict]:
        """
        Constructs a ChromaDB-compliant metadata filter dictionary.
        Supports single filters or multi-logical operators ($and).
        """
        filters = []
        if department:
            filters.append({"department": department})
        if capacity_status:
            filters.append({"capacity_status": capacity_status})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    def retrieve_relevant_faculty(
        self,
        query: str,
        k: int = 5,
        department: Optional[str] = None,
        capacity_status: Optional[str] = None
    ) -> FacultyRetrievalResult:
        """
        Performs semantic similarity search on the faculty dataset.
        
        Args:
            query (str): The search term or user question.
            k (int): Number of matches to retrieve.
            department (Optional[str]): Optional metadata filter for department.
            capacity_status (Optional[str]): Optional metadata filter for capacity status.
            
        Returns:
            FacultyRetrievalResult: The structured outcome containing matches or an error message.
        """
        # Validate query
        if not query or not query.strip():
            logger.warning("Empty search query received.")
            return FacultyRetrievalResult(
                success=False,
                error_message="Invalid query: Search query cannot be empty."
            )

        # Validate Chroma client
        if not self.client:
            return FacultyRetrievalResult(
                success=False,
                error_message="ChromaDB client is uninitialized."
            )

        # 1. Connect/retrieve collection and verify database is not empty
        try:
            # Check if collections exist
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                return FacultyRetrievalResult(
                    success=False,
                    error_message=f"Collection '{self.collection_name}' not found. Please run the ingestion pipeline first."
                )

            collection = self.client.get_collection(name=self.collection_name)
            total_elements = collection.count()
            if total_elements == 0:
                return FacultyRetrievalResult(
                    success=False,
                    error_message="ChromaDB collection is empty. No faculty profiles have been ingested."
                )
        except Exception as e:
            logger.error(f"Error accessing collection: {e}")
            return FacultyRetrievalResult(
                success=False,
                error_message=f"Database access failure: {str(e)}"
            )

        # 2. Embed query using same SentenceTransformer model
        try:
            from ingestion.embed_documents import embed_documents
            query_embeddings = embed_documents([query])
            if not query_embeddings:
                return FacultyRetrievalResult(
                    success=False,
                    error_message="Embedding generation failed to produce vector for query."
                )
            query_vector = query_embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return FacultyRetrievalResult(
                success=False,
                error_message=f"Query embedding failed: {str(e)}"
            )

        # 3. Query ChromaDB with optional metadata filtering
        where_clause = self._build_where_clause(department, capacity_status)
        
        try:
            # Cap k to the total number of records available in collection to prevent out of bounds
            query_k = min(k, total_elements)
            
            logger.info(f"Querying ChromaDB for: '{query}' (k={query_k}, where={where_clause})")
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=query_k,
                where=where_clause
            )
        except Exception as e:
            logger.error(f"ChromaDB query operation failed: {e}")
            return FacultyRetrievalResult(
                success=False,
                error_message=f"ChromaDB query execution failed: {str(e)}"
            )

        # 4. Process matches and fetch full profiles
        try:
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            if not ids:
                logger.info("No matching faculty profiles found.")
                return FacultyRetrievalResult(
                    matches=[],
                    success=True,
                    error_message=None
                )

            matches = []
            for idx, faculty_id in enumerate(ids):
                # Construct file path to load full profile
                file_path = os.path.join("data", "faculty", f"{faculty_id}.json")
                if not os.path.exists(file_path):
                    logger.warning(f"Faculty file not found for ID '{faculty_id}': {file_path}")
                    continue

                profile = load_faculty_profile(file_path)
                if not profile:
                    logger.warning(f"Failed to load/validate profile for ID '{faculty_id}' from {file_path}")
                    continue

                # Compute similarity score from Chroma distance metric (Squared L2)
                # Similarity = 1.0 / (1.0 + distance)
                dist = distances[idx] if idx < len(distances) else 0.0
                similarity_score = 1.0 / (1.0 + dist)

                match = FacultyMatch.from_profile(profile, similarity_score)
                matches.append(match)

            return FacultyRetrievalResult(
                matches=matches,
                success=True,
                error_message=None
            )

        except Exception as e:
            logger.error(f"Error formulating retrieval results: {e}")
            return FacultyRetrievalResult(
                success=False,
                error_message=f"Result post-processing failure: {str(e)}"
            )


def faculty_retrieval_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    Agent node responsible for retrieving and ranking faculty members using RAG.
    Preserved interface for Step 2 LangGraph integration.
    """
    print("[Agent Node] executing faculty_retrieval_agent_node...")
    query = state.get("current_query", "")
    
    # Initialize FacultyRetrievalAgent
    agent = FacultyRetrievalAgent()
    result = agent.retrieve_relevant_faculty(query=query)
    
    if result.success and result.matches:
        # Convert matches to dict for state compatibility if needed
        serialized_matches = [m.model_dump() for m in result.matches]
        return {
            "faculty_results": serialized_matches,
            "error": None
        }
    else:
        return {
            "faculty_results": [],
            "error": result.error_message or "No matches found."
        }

