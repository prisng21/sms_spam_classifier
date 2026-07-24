import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer

# Render (and most non-Heroku hosts) don't auto-install NLTK data from
# nltk.txt the way Heroku's buildpack does, so download it here instead.
# quiet=True keeps this from spamming the Streamlit logs on every restart;
# NLTK skips the download automatically if the data is already present.
for _resource in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(_resource, quiet=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [ps.stem(t) for t in tokens if len(t) > 1]
    return " ".join(tokens)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(
    page_title="Message Inspector",
    page_icon="\u2709\ufe0f",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Design: postal-inspection theme. Every SMS/email is treated like a piece
# of mail passing through a sorting office -- it gets stamped either
# DELIVERED (safe) or QUARANTINED (spam). Grounds the abstract ML output in
# something concrete and specific to what the app actually does.
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink:        #12161d;
    --panel:      #1b212b;
    --panel-line: #333c4a;
    --parchment:  #e9e2cd;
    --parchment-dim: #a9a48f;
    --safe:       #5c8a63;
    --safe-glow:  rgba(92,138,99,0.12);
    --danger:     #b8453c;
    --danger-glow: rgba(184,69,60,0.14);
}

html, body, [class*="css"]  { font-family: 'IBM Plex Mono', monospace; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, #1a2029 0%, var(--ink) 55%);
    color: var(--parchment);
}

/* Hide default streamlit chrome we don't want */
#MainMenu, footer, header {visibility: hidden;}

.masthead {
    border: 1px solid var(--panel-line);
    border-bottom: 3px double var(--panel-line);
    padding: 22px 26px 16px 26px;
    margin-bottom: 28px;
    background: var(--panel);
}
.masthead .kicker {
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--parchment-dim);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.masthead h1 {
    font-family: 'Special Elite', monospace;
    font-size: 30px;
    letter-spacing: 1px;
    margin: 0 0 6px 0;
    color: var(--parchment);
}
.masthead p {
    font-size: 13px;
    color: var(--parchment-dim);
    margin: 0;
    line-height: 1.5;
}

.stTextArea textarea {
    background: #0e1116 !important;
    color: var(--parchment) !important;
    border: 1px solid var(--panel-line) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: var(--parchment-dim) !important;
    box-shadow: none !important;
}
.stTextArea label {
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
    color: var(--parchment-dim) !important;
}

div.stButton > button {
    background: var(--parchment);
    color: var(--ink);
    border: none;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 13px;
    padding: 10px 22px;
    width: 100%;
    transition: transform 0.08s ease, background 0.15s ease;
}
div.stButton > button:hover {
    background: #fff;
    transform: translateY(-1px);
}
div.stButton > button:active { transform: translateY(0px); }

.chip-row button {
    background: transparent !important;
    color: var(--parchment-dim) !important;
    border: 1px dashed var(--panel-line) !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: none !important;
    padding: 6px 10px !important;
    font-weight: 400 !important;
}
.chip-row button:hover {
    color: var(--parchment) !important;
    border-color: var(--parchment-dim) !important;
}

.stamp-wrap {
    margin-top: 28px;
    border: 1px solid var(--panel-line);
    background: var(--panel);
    padding: 30px 26px;
    position: relative;
    overflow: hidden;
}
.stamp-wrap.safe    { box-shadow: inset 4px 0 0 var(--safe); }
.stamp-wrap.danger  { box-shadow: inset 4px 0 0 var(--danger); }

.stamp-label {
    display: inline-block;
    font-family: 'Special Elite', monospace;
    font-size: 26px;
    letter-spacing: 4px;
    padding: 6px 18px;
    border: 3px solid currentColor;
    transform: rotate(-3deg);
    text-transform: uppercase;
}
.stamp-label.safe   { color: var(--safe); }
.stamp-label.danger { color: var(--danger); }

.stamp-sub {
    margin-top: 14px;
    font-size: 12px;
    color: var(--parchment-dim);
    letter-spacing: 1px;
}
.stamp-sub b { color: var(--parchment); }

.confidence-track {
    margin-top: 12px;
    height: 5px;
    width: 100%;
    background: #0e1116;
    border: 1px solid var(--panel-line);
}
.confidence-fill.safe   { background: var(--safe); height: 100%; }
.confidence-fill.danger { background: var(--danger); height: 100%; }

.sidebar-note {
    font-size: 12px;
    color: var(--parchment-dim);
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar -----------------------------------------------------------
with st.sidebar:
    st.markdown("### \u2709\ufe0f How this works")
    st.markdown(
        """<div class="sidebar-note">
        Your message is stemmed and vectorized (TF&#8209;IDF), then scored by a
        Logistic Regression model trained on labeled SMS/email data.<br><br>
        It flags patterns common in spam &mdash; urgency, prize language,
        suspicious links &mdash; not any single keyword.<br><br>
        This is a demo classifier. Always use your own judgment on real
        messages, especially ones asking for money or personal info.
        </div>""",
        unsafe_allow_html=True,
    )

# --- Header --------------------------------------------------------------
st.markdown("""
<div class="masthead">
    <div class="kicker">Sorting Office &middot; Automated Triage</div>
    <h1>Message Inspector</h1>
    <p>Paste a text or email below. It gets read, stamped, and routed &mdash;
    same as anything else that comes through the office.</p>
</div>
""", unsafe_allow_html=True)

# --- Example chips ---------------------------------------------------------
if "draft" not in st.session_state:
    st.session_state.draft = ""

examples = {
    "Prize scam": "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now before it expires!",
    "Bank alert": "Your account has been suspended. Verify your identity immediately at the link below to avoid closure.",
    "Casual text": "Hey, are we still on for dinner tonight? Let me know what time works.",
}
st.markdown('<div class="chip-row">', unsafe_allow_html=True)
chip_cols = st.columns(len(examples))
for col, (label, text) in zip(chip_cols, examples.items()):
    with col:
        if st.button(label, key=f"chip_{label}", use_container_width=True):
            st.session_state.draft = text
st.markdown('</div>', unsafe_allow_html=True)

# --- Input -----------------------------------------------------------------
input_sms = st.text_area(
    "Message to inspect",
    value=st.session_state.draft,
    height=130,
    placeholder="Paste the SMS or email text here...",
    key="draft",
)

inspect = st.button("Inspect message")

# --- Result ------------------------------------------------------------
if inspect:
    if not input_sms.strip():
        st.warning("Nothing to inspect yet \u2014 paste a message above first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector_input)[0]
            confidence = proba[int(result)] * 100

        if result == 1:
            css_class, label, verb = "danger", "Quarantined", "held back as likely spam"
        else:
            css_class, label, verb = "safe", "Delivered", "cleared as safe to read"

        conf_html = ""
        if confidence is not None:
            conf_html = f"""
            <div class="stamp-sub">Confidence: <b>{confidence:.1f}%</b></div>
            <div class="confidence-track">
                <div class="confidence-fill {css_class}" style="width:{confidence:.1f}%"></div>
            </div>
            """

        st.markdown(f"""
        <div class="stamp-wrap {css_class}">
            <div class="stamp-label {css_class}">{label}</div>
            <div class="stamp-sub">This message was {verb}.</div>
            {conf_html}
        </div>
        """, unsafe_allow_html=True)
