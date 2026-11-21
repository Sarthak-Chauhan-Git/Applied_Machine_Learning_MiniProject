import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


MODEL_PATH = Path(__file__).resolve().with_name("news_classifier.joblib")
CATEGORIES = [
    "Business",
    "Education",
    "Entertainment",
    "Health",
    "Politics",
    "Science",
    "Sports",
    "Technology",
]
CATEGORY_COLORS = {
    "Business": ("#1D4ED8", "#E0EAFF"),
    "Education": ("#7C3AED", "#EDE4FF"),
    "Entertainment": ("#DB2777", "#FCE4F0"),
    "Health": ("#047857", "#D9F5EA"),
    "Politics": ("#92400E", "#FCEBD6"),
    "Science": ("#0E7490", "#D9F1F7"),
    "Sports": ("#C2410C", "#FFE8D6"),
    "Technology": ("#4338CA", "#E3E1FF"),
}

st.set_page_config(page_title="newsIQ", page_icon="📰", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def predict(text: str):
    model = load_model()
    cleaned = clean_text(text)
    if not cleaned:
        return None, [], cleaned

    predicted = model.predict([cleaned])[0]
    scores = np.atleast_2d(model.decision_function([cleaned]))
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    ranking = sorted(
        zip(list(model.classes_), probs[0]),
        key=lambda item: item[1],
        reverse=True,
    )
    return predicted, ranking, cleaned


# ---------- session state ----------
st.session_state.setdefault("headline", "")
st.session_state.setdefault("article_text", "")
st.session_state.setdefault("result", None)


def use_example():
    st.session_state.headline = "Team wins national sports championship"
    st.session_state.article_text = (
        "The team secured a major victory after a strong performance in the "
        "national championship match."
    )
    st.session_state.result = None


def clear_all():
    st.session_state.headline = ""
    st.session_state.article_text = ""
    st.session_state.result = None


# ---------- styles ----------
st.markdown(
    """
<style>
:root {
    --ink: #14213D;
    --muted: #5C6B7A;
    --border: #DAD5CB;
    --surface: #FFFFFF;
    --canvas: #F7F4EE;
    --red: #C1121F;
    --red-hover: #A10E19;
    --teal: #1F7A6F;
}

/* Page */
[data-testid="stAppViewContainer"] { background: var(--canvas); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], footer { display: none; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }

/* Hero */
.hero { margin-bottom: 1.6rem; }
.badge {
    display: inline-block; background: #FBE3E5; color: var(--red-hover);
    font-size: .72rem; font-weight: 800; letter-spacing: .1em;
    text-transform: uppercase; padding: .28rem .7rem; border-radius: 999px;
}
.app-title {
    font-size: 2.9rem; font-weight: 800; color: var(--ink);
    letter-spacing: -.03em; line-height: 1.1; margin: .55rem 0 .25rem;
}
.app-title span { color: var(--red); }
.app-subtitle { color: var(--muted); font-size: 1.05rem; }

/* Panels (real containers, so widgets sit INSIDE the card) */
.st-key-input_panel, .st-key-result_panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem 1.6rem 1.6rem;
    box-shadow: 0 12px 30px rgba(20, 33, 61, 0.07);
}
.panel-title {
    color: var(--ink); font-size: 1.15rem; font-weight: 800; margin-bottom: .2rem;
}
.panel-hint { color: var(--muted); font-size: .9rem; margin-bottom: .6rem; }

/* Labels */
label[data-testid="stWidgetLabel"] p {
    color: var(--ink) !important; font-weight: 700 !important; font-size: .92rem;
}

/* Inputs: style the WRAPPER once, make the inner element borderless */
div[data-baseweb="input"], div[data-baseweb="textarea"] {
    background: #FBFAF7 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
    transition: border-color .15s, box-shadow .15s, background .15s;
}
div[data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    background: #FFFFFF !important;
    border-color: var(--ink) !important;
    box-shadow: 0 0 0 3px rgba(20, 33, 61, 0.10) !important;
}
.stTextInput input, .stTextArea textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    font-size: .98rem;
    padding: .75rem .9rem !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton > button {
    height: 48px; border-radius: 12px;
    border: 1.5px solid var(--border);
    background: var(--surface); color: var(--ink);
    font-weight: 700; transition: all .15s;
}
.stButton > button:hover {
    border-color: var(--ink); color: var(--ink); background: #FBFAF7;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    border: none; background: var(--red); color: #fff;
    box-shadow: 0 10px 20px rgba(193, 18, 31, 0.25);
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--red-hover); color: #fff;
}
.stButton > button[kind="primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] p { color: #fff !important; }

/* Chips */
.chip {
    display: inline-block; border-radius: 999px;
    padding: .38rem .8rem; margin: .22rem .3rem .22rem 0;
    font-size: .8rem; font-weight: 700;
}

/* Empty-state placeholder */
.placeholder {
    border: 2px dashed var(--border); border-radius: 14px;
    background: repeating-linear-gradient(135deg, #FBFAF7, #FBFAF7 10px, #F7F4EE 10px, #F7F4EE 20px);
    padding: 2rem 1.2rem; text-align: center; margin: .4rem 0 1.1rem;
}
.placeholder .icon { font-size: 2rem; }
.placeholder .big { color: var(--ink); font-weight: 800; margin-top: .3rem; }
.placeholder .small { color: var(--muted); font-size: .9rem; margin-top: .15rem; }
.section-label {
    color: var(--muted); font-size: .75rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: .09em; margin: .4rem 0 .3rem;
}

/* Result card */
.result-card {
    border: 1px solid var(--border); border-radius: 14px;
    padding: 1.2rem 1.3rem; margin: .4rem 0 1.2rem; background: #FBFAF7;
}
.predicted-chip {
    display: inline-block; border-radius: 999px;
    padding: .5rem 1rem; margin-top: .45rem;
    font-size: 1.35rem; font-weight: 800;
}
.confidence { color: var(--teal); font-weight: 800; margin-top: .7rem; }

/* Top-3 bars */
.bar-row { margin: .65rem 0; }
.bar-head {
    display: flex; justify-content: space-between;
    color: var(--ink); font-weight: 700; font-size: .9rem; margin-bottom: .25rem;
}
.bar-track { background: #EEE9DF; border-radius: 999px; height: 10px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- header ----------
st.markdown(
    '<div class="hero">'
    '<span class="badge">AI news classifier</span>'
    '<div class="app-title">news<span>IQ</span></div>'
    '<div class="app-subtitle">Enter a headline and article text to predict its news category.</div>'
    "</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 1], gap="large")

# ---------- input panel ----------
with left:
    with st.container(key="input_panel"):
        st.markdown(
            '<div class="panel-title">Your article</div>'
            '<div class="panel-hint">Add a headline and the body text for the best accuracy.</div>',
            unsafe_allow_html=True,
        )
        title = st.text_input(
            "Article headline",
            key="headline",
            placeholder="Example: Central bank announces new policy",
        )
        article = st.text_area(
            "Article text",
            key="article_text",
            height=220,
            placeholder="Paste or write the news article here...",
        )

        c1, c2 = st.columns(2)
        c1.button("Use an example", on_click=use_example, use_container_width=True)
        c2.button("Clear", on_click=clear_all, use_container_width=True)

        classify = st.button("Classify news", type="primary", use_container_width=True)

# ---------- classify ----------
if classify:
    full_text = " ".join(p.strip() for p in (title, article) if p.strip())
    if not full_text:
        st.session_state.result = {"error": "Please enter a headline or article text."}
    else:
        predicted, ranking, _ = predict(full_text)
        if predicted is None:
            st.session_state.result = {
                "error": "The text does not contain enough usable words."
            }
        else:
            st.session_state.result = {"predicted": predicted, "ranking": ranking}


def chips_html(names):
    return "".join(
        f'<span class="chip" style="color:{CATEGORY_COLORS[n][0]};'
        f'background:{CATEGORY_COLORS[n][1]};">{n}</span>'
        for n in names
    )


# ---------- result panel ----------
with right:
    with st.container(key="result_panel"):
        st.markdown('<div class="panel-title">Prediction</div>', unsafe_allow_html=True)
        result = st.session_state.result

        if result is None:
            st.markdown(
                '<div class="placeholder">'
                '<div class="icon">📰</div>'
                '<div class="big">No result yet</div>'
                '<div class="small">Your classification result will appear here.</div>'
                "</div>"
                '<div class="section-label">Categories I can detect</div>'
                + chips_html(CATEGORIES),
                unsafe_allow_html=True,
            )
        elif "error" in result:
            st.warning(result["error"])
        else:
            predicted = result["predicted"]
            ranking = result["ranking"]
            confidence = ranking[0][1] if ranking else 0
            fg, bg = CATEGORY_COLORS[predicted]

            st.markdown(
                '<div class="result-card">'
                '<div class="section-label">Predicted category</div>'
                f'<span class="predicted-chip" style="color:{fg};background:{bg};">{predicted}</span>'
                f'<div class="confidence">{confidence:.1%} relative confidence</div>'
                "</div>",
                unsafe_allow_html=True,
            )

            bars = '<div class="section-label">Top categories</div>'
            for category, score in ranking[:3]:
                cfg, _ = CATEGORY_COLORS[category]
                bars += (
                    '<div class="bar-row">'
                    f'<div class="bar-head"><span>{category}</span><span>{score:.1%}</span></div>'
                    '<div class="bar-track">'
                    f'<div class="bar-fill" style="width:{score * 100:.1f}%;background:{cfg};"></div>'
                    "</div></div>"
                )
            st.markdown(bars, unsafe_allow_html=True)
