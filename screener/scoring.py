from dataclasses import dataclass, field

from .config import ScoringConfig
from .embedder import Embedder
from .extractor import EDU_LABELS, JobProfile, ResumeProfile, tokenize

STRONG, PARTIAL = 0.65, 0.35


@dataclass
class RequirementMatch:
    requirement: str
    nice: bool
    strength: float          
    evidence: str            

    @property
    def label(self) -> str:
        return "Strong" if self.strength >= STRONG else "Partial" if self.strength >= PARTIAL else "Weak"


@dataclass
class ScreeningResult:
    name: str
    score: float = 0.0
    components: dict[str, float | None] = field(default_factory=dict)
    matches: list[RequirementMatch] = field(default_factory=list)
    years: float = 0.0
    edu_label: str = ""
    reasons: list[str] = field(default_factory=list)
    error: str | None = None


def match_requirements(resume: ResumeProfile, job: JobProfile, req_vecs, emb: Embedder) -> list[RequirementMatch]:
    reqs = [(t, False) for t in job.must] + [(t, True) for t in job.nice]
    if not reqs:
        return []
    sims = req_vecs @ emb.encode(resume.units).T                    # requirements x resume lines
    matches = []
    for i, (text, nice) in enumerate(reqs):
        j = int(sims[i].argmax())
        words = tokenize(text)
        overlap = len(words & resume.tokens) / len(words) if words else 0.0
        strength = 0.65 * emb.scale_requirement(float(sims[i, j])) + 0.35 * overlap
        matches.append(RequirementMatch(text, nice, min(strength, 1.0), resume.units[j][:140]))
    return matches


def education_score(level: int, required: int) -> float:
    return 1.0 if level >= required else max(0.0, 1.0 - 0.5 * (required - level))


def score_resume(resume: ResumeProfile, job: JobProfile, job_vec, req_vecs, emb: Embedder,
                 cfg: ScoringConfig) -> ScreeningResult:
    matches = match_requirements(resume, job, req_vecs, emb)
    weights = [cfg.nice_weight if m.nice else 1.0 for m in matches]

    comp: dict[str, float | None] = {
        "skills": sum(w * m.strength for w, m in zip(weights, matches)) / sum(weights) if matches else None,
        "experience": min(resume.years / job.min_years, 1.0) if job.min_years else None,
        "education": education_score(resume.edu_level, job.min_edu) if job.min_edu else None,
        "semantic": emb.similarity(emb.doc_vector(resume.text), job_vec),
    }
    w = {"skills": cfg.w_skills, "experience": cfg.w_experience,
         "education": cfg.w_education, "semantic": cfg.w_semantic}
    active = {k: v for k, v in comp.items() if v is not None}
    score = sum(v * w[k] for k, v in active.items()) / sum(w[k] for k in active)

    reasons = []
    if matches:
        counts = {lab: sum(m.label == lab for m in matches) for lab in ("Strong", "Partial", "Weak")}
        reasons.append(f"Requirements: {counts['Strong']} strong, {counts['Partial']} partial, "
                       f"{counts['Weak']} weak out of {len(matches)}")
    if job.min_years:
        reasons.append(f"Experience: {resume.years:g} yrs vs {job.min_years:g} required")
    if job.min_edu:
        reasons.append(f"Education: {EDU_LABELS[resume.edu_level]} vs {EDU_LABELS[job.min_edu]} required")
    reasons.append(f"Overall semantic fit with job description: {comp['semantic']:.0%}")
    return ScreeningResult(resume.name, score, comp, matches, resume.years, EDU_LABELS[resume.edu_level], reasons)


def decide(result: ScreeningResult, cfg: ScoringConfig) -> bool:
    """Selection rule: the final score must reach the threshold."""
    return result.error is None and result.score >= cfg.threshold
