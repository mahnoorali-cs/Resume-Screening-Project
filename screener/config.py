from dataclasses import dataclass

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RESUMES = 10


@dataclass
class ScoringConfig:
    w_skills: float = 0.40     
    w_experience: float = 0.20
    w_education: float = 0.15
    w_semantic: float = 0.25     
    threshold: float = 0.60      
    nice_weight: float = 0.5     
