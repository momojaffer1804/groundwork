import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Groundwork",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---- Custom styling ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0f1117;
    }

    .block-container {
        padding-top: 3rem;
        max-width: 780px;
    }

    .groundwork-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f5f5f7;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .groundwork-subtitle {
        font-size: 1rem;
        color: #9a9aa5;
        margin-bottom: 2.2rem;
        font-weight: 400;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background-color: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        color: #f5f5f7;
        padding: 0.7rem 0.9rem;
        font-size: 0.95rem;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 1px #6366f1;
    }

    .stButton button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.15s ease;
    }

    .stButton button:hover {
        opacity: 0.88;
        color: white;
    }

    .answer-card {
        background-color: #161922;
        border: 1px solid #262a38;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.6rem;
    }

    .answer-card.refused {
        border-color: #3a2a2a;
        background-color: #1c1516;
    }

    .answer-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7a7d8a;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .answer-text {
        font-size: 1.15rem;
        color: #f5f5f7;
        font-weight: 500;
        line-height: 1.5;
    }

    .refused-text {
        font-size: 1.05rem;
        color: #e08a8a;
        font-weight: 500;
        line-height: 1.5;
    }

    .confidence-row {
        margin-top: 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .confidence-label {
        font-size: 0.85rem;
        color: #7a7d8a;
    }

    .confidence-value {
        font-size: 0.85rem;
        color: #a8a8b3;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="groundwork-title">Groundwork</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="groundwork-subtitle">Grounded QA over research papers. '
    'Answers only from the paper itself, refuses when it can\'t find a reliable answer.</div>',
    unsafe_allow_html=True,
)

# ---- Inputs ----
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

question = st.text_area(
    "Question",
    placeholder="e.g. What optimizer was used for training?",
    height=90,
)

ask_clicked = st.button("Ask", use_container_width=False)

# ---- Handle request ----
if ask_clicked:
    if uploaded_file is None:
        st.warning("Upload a PDF first.")
    elif not question.strip():
        st.warning("Enter a question first.")
    else:
        cache_key = f"paper_id::{uploaded_file.name}::{uploaded_file.size}"

        if cache_key not in st.session_state:
            with st.spinner("Parsing paper..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    upload_response = requests.post(f"{API_URL}/upload", files=files, timeout=180)
                    upload_response.raise_for_status()
                    st.session_state[cache_key] = upload_response.json()["paper_id"]
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the API. Make sure it's running: `uvicorn api.main:app --reload`")
                    st.stop()
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.stop()

        paper_id = st.session_state[cache_key]

        with st.spinner("Retrieving, reranking, reading..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"paper_id": paper_id, "question": question},
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Can't reach the API. Make sure it's running: "
                    "`uvicorn api.main:app --reload`"
                )
                data = None
            except Exception as e:
                st.error(f"Request failed: {e}")
                data = None

        if data:
            if data["refused"]:
                st.markdown(f"""
                <div class="answer-card refused">
                    <div class="answer-label">Refused</div>
                    <div class="refused-text">{data['reason']}</div>
                    <div class="confidence-row">
                        <span class="confidence-label">Confidence</span>
                        <span class="confidence-value">{data['confidence']:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-label">Answer</div>
                    <div class="answer-text">{data['answer']}</div>
                    <div class="confidence-row">
                        <span class="confidence-label">Confidence</span>
                        <span class="confidence-value">{data['confidence']:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)