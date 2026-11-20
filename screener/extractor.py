import re
from dataclasses import dataclass
from datetime import date

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_BULLET = re.compile(r"^[\s\-•*·▪●–—>]+")

# ------------------------------------------------------------------------ keywords
_GENERIC = {"experience", "experienced", "strong", "knowledge", "skills", "skill", "ability", "proven",
            "good", "excellent", "understanding", "familiarity", "proficient", "solid", "years", "year",
            "related", "field", "plus", "preferred", "required", "including", "etc", "well", "working", "using"}
_STOP = set(ENGLISH_STOP_WORDS) | _GENERIC


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[:-len(suf)]
    return t


def tokenize(text: str) -> set[str]:
    words = (w.rstrip(".") for w in re.findall(r"[a-z0-9][a-z0-9+#.]*", text.lower()))
    return {_stem(w) for w in words if len(w) > 1 and not w.isdigit() and w not in _STOP}


# education
EDU_LABELS = {0: "Not specified", 1: "High school", 2: "Diploma", 3: "Bachelor's", 4: "Master's", 5: "PhD"}
_EDU_PATTERNS = [
    (5, r"\b(ph\.?\s?d|doctorate|doctoral)\b"),
    (4, r"\b(master'?s?|m\.?sc|m\.?phil|mba|m\.?tech|m\.?eng)\b"
        r"|(?-i:\bMS(?:CS|SE|IT|AI)?\b)(?!\s+(?:office|word|excel|powerpoint|sql|teams|access))"),
    (3, r"\b(bachelor'?s?|b\.?sc|b\.?tech|b\.?eng|bba|bca)\b|(?-i:\bBS(?:CS|SE|IT|AI)?\b)"),
    (2, r"\b(diploma|associate'?s? degree|dae)\b"),
    (1, r"\b(f\.?sc|a[- ]levels?|high school|matric(?:ulation)?|hssc)\b"),
]
_EDU_RES = [(lvl, re.compile(p, re.I)) for lvl, p in _EDU_PATTERNS]


def detect_education_levels(text: str) -> list[int]:
    return sorted({lvl for lvl, rx in _EDU_RES if rx.search(text)})


# experience
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_RANGE = re.compile(
    r"(?:(?P<m1>[A-Za-z]{3,9})\.?,?\s+)?(?P<y1>(?:19|20)\d{2})\s*(?:-|–|—|to)\s*"
    r"(?:(?:(?P<m2>[A-Za-z]{3,9})\.?,?\s+)?(?P<y2>(?:19|20)\d{2})|(?P<now>present|current|now|ongoing|date))",
    re.I)
_STATED = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[\w\-]+\s+){0,3}experience", re.I)
_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)", re.I)


def _month_index(year: int, month_name: str | None) -> int:
    month = _MONTHS.get((month_name or "")[:3].lower(), 6)  
    return year * 12 + month


def experience_years(text: str) -> float:
    today = date.today()
    spans = []
    for m in _RANGE.finditer(text):
        start = _month_index(int(m["y1"]), m["m1"])
        end = today.year * 12 + today.month if m["now"] else _month_index(int(m["y2"]), m["m2"])
        if end >= start:
            spans.append((start, end))
    total, last_end = 0, None
    for start, end in sorted(spans):
        if last_end is None or start > last_end:
            total += end - start
            last_end = end
        elif end > last_end:
            total += end - last_end
            last_end = end
    return total / 12


def stated_years(text: str) -> float:
    return max((float(x) for x in _STATED.findall(text)), default=0.0)


# sections
_SECTIONS = {
    "education": r"education|academic(?:s| background| qualifications?)?",
    "experience": r"(?:work |professional |employment |relevant )?(?:experience|history)|internships?",
    "skills": r"(?:technical |key |core )?skills|technologies|expertise",
    "projects": r"(?:academic |personal )?projects?",
}
_SECTION_RES = {k: re.compile(v, re.I) for k, v in _SECTIONS.items()}


def split_sections(text: str) -> dict[str, str]:
    sections, current = {"header": []}, "header"
    for line in text.splitlines():
        title = line.strip().rstrip(":").strip()
        key = next((k for k, rx in _SECTION_RES.items()
                    if title and len(title.split()) <= 4 and rx.fullmatch(title)), None)
        if key:
            current = key
            sections.setdefault(key, [])
        else:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# profiles
@dataclass
class ResumeProfile:
    name: str
    text: str
    tokens: set[str]      
    units: list[str]        
    edu_level: int
    years: float


@dataclass
class JobProfile:
    text: str
    must: list[str]      
    nice: list[str]        
    min_edu: int
    min_years: float


def parse_resume(name: str, text: str) -> ResumeProfile:
    sections = split_sections(text)
    levels = detect_education_levels(sections.get("education", "")) or detect_education_levels(text)
    years = max(experience_years(sections.get("experience", "")), stated_years(text))
    lines = [l for l in (_BULLET.sub("", l).strip() for l in text.splitlines()) if len(l.split()) >= 2][:200]
    units = lines + [f"{a} {b}" for a, b in zip(lines, lines[1:])]
    return ResumeProfile(name, text, tokenize(text), units, max(levels, default=0), round(years, 1))


_NICE = re.compile(r"preferred|nice[\s-]to[\s-]have|good to have|bonus|\bplus\b|desirable|optional|advantage", re.I)
_MUST_HDR = re.compile(r"requirements?|qualifications?|responsibilities|must[\s-]have|required", re.I)


def parse_job(text: str) -> JobProfile:
    """Split a job description into required / nice-to-have lines + education & experience limits."""
    rows = [(raw, _BULLET.sub("", raw).strip(), bool(_BULLET.match(raw))) for raw in text.splitlines()]
    rows = [(raw, line, bullet) for raw, line, bullet in rows if line]

    def is_header(line, bullet):
        words = len(line.split())
        return not bullet and ((line.endswith(":") and words <= 8) or words <= 4)

    has_headers = any(is_header(l, b) and (_NICE.search(l) or _MUST_HDR.search(l)) for _, l, b in rows)
    mode, must, nice, required_lines = ("intro" if has_headers else "must"), [], [], []
    for _, line, bullet in rows:
        if is_header(line, bullet) and _NICE.search(line):
            mode = "nice"
        elif is_header(line, bullet) and _MUST_HDR.search(line):
            mode = "must"
        elif mode == "intro":
            continue
        elif mode == "nice" or _NICE.search(line):
            nice.append(line)
        else:
            required_lines.append(line)
            is_edu = bool(detect_education_levels(line))
            is_years = "experience" in line.lower() and bool(_YEARS.search(line))
            if not (is_edu or is_years):          # those are scored separately
                must.append(line)

    levels = detect_education_levels("\n".join(required_lines))
    years = max((float(m[1]) for l in required_lines if "experience" in l.lower()
                 for m in [_YEARS.search(l)] if m), default=0.0)
    return JobProfile(text, must, nice, min(levels, default=0), years)
