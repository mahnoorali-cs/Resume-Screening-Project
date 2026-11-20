import pandas as pd
import streamlit as st

from screener.config import MAX_RESUMES, ScoringConfig
from screener.embedder import Embedder
from screener.extractor import EDU_LABELS
from screener.pipeline import screen_resumes
from screener.scoring import decide

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading language model (first run downloads it)...")
def get_embedder() -> Embedder:
    return Embedder()


# sidebar settings
with st.sidebar:
    st.header("Selection rules")
    threshold = st.slider("Minimum score to select", 0, 100, 60, 5, format="%d%%") / 100
    with st.expander("Scoring weights (apply on next screening)"):
        w_skills = st.slider("Skills", 0, 100, 40)
        w_exp = st.slider("Experience", 0, 100, 20)
        w_edu = st.slider("Education", 0, 100, 15)
        w_sem = st.slider("Semantic fit", 0, 100, 25)
    embedder = get_embedder()
    st.caption(f"Model: {embedder.name}")
    if not embedder.is_deep:
        st.warning("Deep model not available - run `pip install sentence-transformers` "
                   "and check your internet connection for the first download.")

cfg = ScoringConfig(w_skills=w_skills or 1, w_experience=w_exp or 1, w_education=w_edu or 1,
                    w_semantic=w_sem or 1, threshold=threshold)

# inputs
st.title("📄 AI Powered Resume Screening")
left, right = st.columns(2)
jd_text = left.text_area("Job description", height=280,
                         placeholder="Paste the job description: required education, skills, years of experience...")
uploads = right.file_uploader(f"Resumes (PDF or TXT, max {MAX_RESUMES})", type=["pdf", "txt"],
                              accept_multiple_files=True)

if len(uploads) > MAX_RESUMES:
    st.error(f"Please upload at most {MAX_RESUMES} resumes (you added {len(uploads)}).")
elif st.button("Screen resumes", type="primary", disabled=not (jd_text.strip() and uploads)):
    with st.spinner("Screening..."):
        files = [(u.name, u.getvalue()) for u in uploads]
        st.session_state["job"], st.session_state["results"] = screen_resumes(files, jd_text, embedder, cfg)

# results
if "results" in st.session_state:
    job, results = st.session_state["job"], st.session_state["results"]
    for r in results:
        r.selected = decide(r, cfg)
    selected = [r for r in results if r.selected]

    with st.expander("What the system understood from the job description", expanded=True):
        st.write("**Required:**")
        st.markdown("\n".join(f"- {t}" for t in job.must) or "-")
        st.write("**Nice to have:**")
        st.markdown("\n".join(f"- {t}" for t in job.nice) or "-")
        st.write(f"**Minimum education:** {EDU_LABELS[job.min_edu]}  |  "
                 f"**Minimum experience:** {job.min_years:g} years")

    c1, c2 = st.columns(2)
    c1.metric("Resumes screened", len(results))
    c2.metric("Selected", len(selected))

    pct = lambda v: None if v is None else round(v * 100)
    table = pd.DataFrame([{
        "Rank": i, "Resume": r.name,
        "Decision": "Error" if r.error else ("✅ Selected" if r.selected else "❌ Rejected"),
        "Score": pct(r.score), "Requirements": pct(r.components.get("skills")),
        "Experience": pct(r.components.get("experience")), "Education": pct(r.components.get("education")),
        "Semantic": pct(r.components.get("semantic")),
    } for i, r in enumerate(results, 1)])
    bar = lambda: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={c: bar() for c in ["Score", "Requirements", "Experience", "Education", "Semantic"]})
    st.download_button("Download selected as CSV", table[table["Decision"] == "✅ Selected"].to_csv(index=False),
                       "selected_resumes.csv", disabled=not selected)

    for r in results:
        with st.expander(f"{r.name}"):
            if r.error:
                st.error(r.error)
                continue
            st.write(f"**Score {r.score:.0%}**  |  {r.years:g} yrs experience  |  {r.edu_label}")
            st.markdown("\n".join(f"- {line}" for line in r.reasons))
            if r.matches:
                st.dataframe(pd.DataFrame([{
                    "Requirement": m.requirement, "Type": "Nice to have" if m.nice else "Required",
                    "Match": m.label, "Closest line in resume": m.evidence} for m in r.matches]),
                    hide_index=True, use_container_width=True)
