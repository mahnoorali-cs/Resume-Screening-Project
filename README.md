# AI Powered Resume Screening

Upload up to 10 resumes (PDF/TXT), paste a job description, and get a ranked, explainable shortlist.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads Sentence-BERT (all-MiniLM-L6-v2, ~90 MB). Without it the app falls back to a
simple offline similarity so it still runs, but results are better with the deep model.

## Pipeline
| Module | Job |
|---|---|
| `screener/pdf_parser.py` | PDF/TXT -> clean text (PyMuPDF) |
| `screener/extractor.py` | education level, years of experience, required / nice-to-have lines of the job description |
| `screener/embedder.py` | Sentence-BERT embeddings (deep learning) |
| `screener/scoring.py` | requirement matching + experience + education + semantic fit -> score, reasons, select/reject |
| `screener/pipeline.py` | ties everything together |
| `app.py` | Streamlit UI |

Skills are not matched from a fixed list: each job requirement line is compared with the closest line of the resume.
Tune weights/thresholds in `screener/config.py`.

terminal-> python -m streamlit run app.py
