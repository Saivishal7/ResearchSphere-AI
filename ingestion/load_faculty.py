# ingestion/load_faculty.py
import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

class Project(BaseModel):
    title: str
    description: str

class Publication(BaseModel):
    title: str
    year: int
    conference_or_journal: str
    abstract: str
    citation_count: int

class FacultyProfile(BaseModel):
    faculty_id: str
    name: str
    designation: str
    department: str
    email: str
    office: str
    research_interests: List[str]
    profile_summary: str
    current_projects: List[Project]
    publications: List[Publication]
    keywords: List[str]
    capacity_status: str

def load_faculty_profile(file_path: str) -> Optional[FacultyProfile]:
    """
    Loads and validates a single faculty JSON profile.
    Returns None if validation fails or file is invalid.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Robust validation using Pydantic
        profile = FacultyProfile(**data)
        return profile
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[Warning] Failed to validate schema or load JSON for {file_path}: {e}")
        return None
    except Exception as e:
        print(f"[Warning] Unexpected error loading {file_path}: {e}")
        return None

def load_all_faculty_profiles(directory_path: str = "data/faculty") -> List[FacultyProfile]:
    """
    Iterates over the given directory and loads all valid faculty JSON files.
    """
    profiles = []
    if not os.path.exists(directory_path):
        print(f"[Error] Directory not found: {directory_path}")
        return []

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            profile = load_faculty_profile(file_path)
            if profile:
                profiles.append(profile)
            else:
                print(f"[Info] Skipping invalid profile file: {filename}")
                
    return profiles

if __name__ == "__main__":
    # Test loading profiles
    print("Testing faculty profiles loader...")
    profiles = load_all_faculty_profiles()
    print(f"Loaded {len(profiles)} faculty profiles successfully.")
