from .config import MAX_RESUMES, ScoringConfig
from .embedder import Embedder
from .extractor import JobProfile, parse_job, parse_resume
from .pdf_parser import ResumeReadError, extract_text
from .scoring import ScreeningResult, score_resume


def screen_resumes(files: list[tuple[str, bytes]], jd_text: str, embedder: Embedder,
                   cfg: ScoringConfig | None = None) -> tuple[JobProfile, list[ScreeningResult]]:
    """`files` is a list of (filename, raw bytes). Returns the parsed job and ranked results."""
    if len(files) > MAX_RESUMES:
        raise ValueError(f"At most {MAX_RESUMES} resumes can be screened at a time.")
    cfg = cfg or ScoringConfig()
    job = parse_job(jd_text)
    job_vec = embedder.doc_vector(jd_text)
    requirements = job.must + job.nice
    req_vecs = embedder.encode(requirements) if requirements else None   # embedded once, reused

    results = []
    for name, data in files:
        try:
            profile = parse_resume(name, extract_text(name, data))
            results.append(score_resume(profile, job, job_vec, req_vecs, embedder, cfg))
        except ResumeReadError as exc:
            results.append(ScreeningResult(name=name, error=str(exc)))

    results.sort(key=lambda r: (r.error is not None, -r.score))
    return job, results
