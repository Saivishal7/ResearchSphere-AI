# ingestion/embed_documents.py
from typing import List
from ingestion.load_faculty import FacultyProfile
from sentence_transformers import SentenceTransformer

_model = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Lazy-loads and caches the SentenceTransformer model to optimize runtime memory.
    """
    global _model
    if _model is None:
        print(f"[Info] Loading SentenceTransformer model: {model_name}...")
        _model = SentenceTransformer(model_name)
        print("[Info] Model loaded successfully.")
    return _model

def compile_faculty_document(profile: FacultyProfile) -> str:
    """
    Intelligently compiles a single, rich semantic document from a faculty profile.
    This includes Name, Department, Research Interests, Profile Summary,
    Current Projects, Publication Titles, and Keywords, while omitting emails and offices.
    """
    interests_str = ", ".join(profile.research_interests)
    keywords_str = ", ".join(profile.keywords)
    
    projects_list = []
    for proj in profile.current_projects:
        projects_list.append(f"- Project: {proj.title}. Description: {proj.description}")
    projects_str = "\n".join(projects_list)
    
    publications_list = []
    for pub in profile.publications:
        publications_list.append(f"- Publication: {pub.title}")
    publications_str = "\n".join(publications_list)
    
    document_text = f"""Faculty Profile
Name: {profile.name}
Department: {profile.department}
Designation: {profile.designation}
Research Interests: {interests_str}
Profile Summary: {profile.profile_summary}

Current Ongoing Projects:
{projects_str}

Key Publication Titles:
{publications_str}

Keywords: {keywords_str}"""
    return document_text.strip()

def embed_documents(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Generates high-quality vector embeddings for a list of compiled documents.
    """
    model = get_embedding_model(model_name)
    # Generate embeddings and convert to native Python float list of lists
    embeddings = model.encode(texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]
