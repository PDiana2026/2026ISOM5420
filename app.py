
"""
ISOM5240 Individual Assignment
Storytelling Application using Hugging Face Pipelines
------------------------------------------------------
Pipeline:
  img2text   →  Salesforce/blip-image-captioning-base
  text2story →  roneneldan/TinyStories-33M   (trained on children's stories)
  text2audio →  kakao-enterprise/vits-ljs

Safety:
  • Taboo-word list guards both caption prompt and generated story output.
  • Story is strictly trimmed / padded to 50-100 words.
  • Up to 3 regeneration attempts before a guaranteed safe fallback is used.

UI Theme: Ocean Adventure — deep teal hero, sunny accent cards, wave details.
"""

import io
import re
import wave
import numpy as np
import streamlit as st
from transformers import pipeline


# ══════════════════════════════════════════════════════════════════════════════
# CHILD-SAFETY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TABOO_WORDS: set = {
    # Violence / harm
    "murder", "kill", "killed", "killing", "kills", "killer",
    "death", "dead", "die", "dying", "dies", "died",
    "blood", "gore", "corpse", "stab", "shoot", "gun", "knife",
    "weapon", "bomb", "explosion", "war", "battle",
    "attack", "abuse", "rape", "terror", "terrorist", "suicide",
    "wound", "injury", "injured",
    # Adult / mature content
    "sex", "sexy", "naked", "nude", "porn",
    "pregnant", "pregnancy",
    # Substances
    "drug", "drugs", "alcohol", "beer", "wine", "vodka",
    "cigarette", "smoke", "smoking",
    # Profanity / hate speech
    "damn", "hell", "ass", "crap", "bastard", "bitch",
    "shit", "fuck", "hate", "evil",
    # Frightening / demeaning
    "devil", "demon", "horror", "ghost", "creepy",
    "idiot", "stupid", "dumb", "ugly",
}

MAX_REGEN_ATTEMPTS: int = 3
MIN_WORDS: int = 50
MAX_WORDS: int = 100

_PADDING = (
    " Together they laughed and played until the golden sun dipped below the hills. "
    "Everyone went home happy. The End."
)

SAFE_FALLBACK_STORY: str = (
    "Once upon a time, a cheerful little bunny named Biscuit lived near a sunny meadow. "
    "Every morning Biscuit hopped through rainbow-coloured flowers, greeting butterflies "
    "and bumble bees. One day he discovered a sparkling pond full of friendly frogs. "
    "They jumped and splashed and sang merry songs all afternoon. When the golden sun "
    "began to set, Biscuit hurried home, where his mum had warm carrot soup waiting. "
    "He snuggled into his cosy bed, dreaming of tomorrow's adventures. The End."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION (Must be the first Streamlit command)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="StoryTeller ✨",
    page_icon="🐠",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CACHING (EFFICIENCY UPGRADE)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Loads heavy AI models into memory once to prevent slow reloads."""
    caption_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    story_model = pipeline(
        "text-generation",
        model="roneneldan/TinyStories-33M",
        max_new_tokens=160,
        temperature=0.75,
        top_p=0.92,
        repetition_penalty=1.3,
        do_sample=True,
    )
    
    tts_model = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
    
    return caption_model, story_model, tts_model

# Load pipelines into global variables
captioner, story_pipe, tts_pipe = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def contains_taboo(text: str) -> bool:
    lowered = text.lower()
    for word in TABOO_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return True
    return False

def enforce_word_count(text: str) -> str:
    def _trim(s: str) -> str:
        words = s.split()
        if len(words) <= MAX_WORDS:
            return s
        candidate = " ".join(words[:MAX_WORDS])
        last = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
        return candidate[: last + 1] if last != -1 else candidate

    text = _trim(text)
    if len(text.split()) < MIN_WORDS:
        text = _trim(text.rstrip() + _PADDING)
    return text

def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    arr = np.squeeze(audio_array).astype(np.float32)
    peak = np.abs(arr).max()
    if peak > 0:
        arr /= peak
    pcm = (arr * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)         # mono
        wf.setsampwidth(2)         # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE FUNCTIONS (Using cached models)
# ══════════════════════════════════════════════════════════════════════════════

def img2text(url: str) -> str:
    return captioner(url)[0]["generated_text"]

def text2story(caption: str) -> tuple:
    safe_caption = (
        caption if not contains_taboo(caption)
        else "a friendly little animal playing in a sunny meadow"
    )
    prompt = f"Once upon a time, {safe_caption}. One happy morning,"

    for _ in range(MAX_REGEN_ATTEMPTS):
        raw = story_pipe(prompt)[0]["generated_text"]
        story = enforce_word_count(raw)
        if not contains_taboo(story):
            return story, False

    return SAFE_FALLBACK_STORY, True

def text2audio(story_text: str) -> dict:
    return tts_pipe(story_text)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Ocean Adventure theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Quicksand:wght@500;600;700&display=swap');

    :root {
        --ocean-deep:  #006d82;
        --ocean-mid:   #0097a7;
        --ocean-light: #e0f7fa;
        --navy:        #01353d;
        --white:       #ffffff;
        --card-r:      18px;
    }

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif !important;
        background: #dff4f8 !important;
    }

    .main > div {
        background:
            repeating-linear-gradient(
                -45deg,
                rgba(255,255,255,0.25) 0px, rgba(255,255,255,0.25) 2px,
                transparent 2px, transparent 18px
            ),
            linear-gradient(180deg, #caf0f8 0%, #ade8f4 40%, #90e0ef 100%);
        min-height: 100vh;
        padding-bottom: 3rem;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0);     }
        50%      { transform: translateY(-10px); }
    }
    @keyframes sway {
        0%, 100% { transform: rotate(-4deg); }
        50%      { transform: rotate( 4deg); }
    }
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.75) translateY(20px); }
        70%  { transform: scale(1.04) translateY(-3px);             }
        100% { opacity: 1; transform: scale(1)    translateY(0);    }
    }

    .hero-creatures {
        display: flex;
        justify-content: center;
        gap: 1.4rem;
        font-size: 2rem;
        margin-bottom: -0.7rem;
        position: relative;
        z-index: 2;
    }
    .hero-creatures span:nth-child(1) { animation: float 3.0s ease-in-out infinite; }
    .hero-creatures span:nth-child(2) { animation: sway  2.4s ease-in-out infinite 0.3s; }
    .hero-creatures span:nth-child(3) { animation: float 2.8s ease-in-out infinite 0.6s; }
    .hero-creatures span:nth-child(4) { animation: sway  3.2s ease-in-out infinite 0.2s; }
    .hero-creatures span:nth-child(5) { animation: float 2.6s ease-in-out infinite 0.9s; }

    .hero-box {
        background: var(--ocean-deep);
        border-radius: 26px 26px 0 0;
        border: 4px solid var(--navy);
        border-bottom: none;
        padding: 1.6rem 1.5rem 1.4rem;
        text-align: center;
        box-shadow: 5px 5px 0 var(--navy);
    }
    .hero-title {
        font-family: 'Baloo 2', cursive !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: var(--white) !important;
        text-shadow: 3px 3px 0 var(--navy);
        margin: 0 0 0.3rem !important;
        line-height: 1.1 !important;
    }
    .hero-sub {
        font-size: 1rem;
        font-weight: 700;
        color: rgba(255,255,255,0.92);
        margin: 0 !important;
    }

    .hero-wave { display: block; width: 100%; overflow: hidden;
                 line-height: 0; margin-bottom: 1.4rem;
                 filter: drop-shadow(5px 4px 0 var(--navy)); }
    .hero-wave svg { display: block; width: 100%; height: 40px; }

    .steps-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin: 0 0 1.8rem;
    }
    .pill {
        border: 2.5px solid var(--navy);
        border-radius: 50px;
        padding: 0.38rem 1rem;
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--navy);
        box-shadow: 3px 3px 0 var(--navy);
        white-space: nowrap;
    }
    .pill-upload { background: var(--ocean-light); }
    .pill-read   { background: #fff9c4; }
    .pill-write  { background: #ede7f6; }
    .pill-hear   { background: #e8f5e9; }
    .arrow-sep   { font-size: 1.2rem; color: var(--navy); font-weight: 900; }

    [data-testid="stFileUploader"] > div > div {
        border: 3px dashed var(--ocean-mid) !important;
        border-radius: 20px !important;
        background: rgba(0,151,167,0.06) !important;
        transition: background 0.2s;
    }
    [data-testid="stFileUploader"] > div > div:hover {
        background: rgba(0,151,167,0.14) !important;
    }

    [data-testid="stImage"] img {
        border-radius: 18px !important;
        border: 4px solid var(--navy) !important;
        box-shadow: 6px 6px 0 var(--navy) !important;
    }

    .card {
        border-radius: var(--card-r);
        border: 3px solid var(--navy);
        padding: 1.3rem 1.6rem;
        margin: 1rem 0;
        box-shadow: 5px 5px 0 var(--navy);
        animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
    }
    .card-caption { background: #fffde7; }
    .card-story   { background: #eef2ff; }
    .card-audio   { background: #e8f5e9; }

    .card-icon {
        font-size: 2rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        margin-bottom: 0.2rem;
    }
    .card-label {
        font-family: 'Baloo 2', cursive;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 0.5rem;
        display: block;
    }
    .card-body {
        font-size: 1.05rem;
        font-weight: 600;
        color: #2a2a2a;
        line-height: 1.8;
    }
    .word-badge {
        display: inline-block;
        background: var(--ocean-deep);
        color: var(--white);
        font-size: 0.7rem;
        font-weight: 700;
        border-radius: 50px;
        padding: 0.12rem 0.65rem;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    .safety-box {
        background: #fff8e1;
        border: 3px solid #ffb300;
        border-radius: 16px;
        padding: 0.85rem 1.2rem;
        font-size: 0.88rem;
        font-weight: 700;
        color: #7a5800;
        box-shadow: 4px 4px 0 #e6ac00;
        margin: 0.6rem 0;
        animation: popIn 0.45s ease both;
    }

    audio {
        width: 100%;
        border-radius: 50px;
        outline: 3px solid var(--navy);
    }

    [data-testid="stAlert"] {
        border-radius: 16px !important;
        border: 2.5px solid var(--navy) !important;
        font-weight: 700 !important;
    }

    hr { border: 2px dashed rgba(0,109,130,0.3) !important; border-radius: 4px; }

    .footer {
        text-align: center;
        color: #7aa8b0;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 2px dashed rgba(0,109,130,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero-creatures">
        <span>🐠</span><span>🐙</span><span>🐬</span><span>🐡</span><span>⭐</span>
    </div>
    <div class="hero-box">
        <div class="hero-title">✨ StoryTeller ✨</div>
        <p class="hero-sub">
            Upload any picture &amp; watch a fairy tale appear — then hear it read aloud! 🎧🌊
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-wave">
        <svg viewBox="0 0 1200 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0,20 C150,40 350,0 600,20 C850,40 1050,0 1200,20 L1200,40 L0,40 Z"
                  fill="#006d82" stroke="#01353d" stroke-width="4"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="steps-row">
        <div class="pill pill-upload">📸 Upload Picture</div>
        <div class="arrow-sep">→</div>
        <div class="pill pill-read">🔍 Read Image</div>
        <div class="arrow-sep">→</div>
        <div class="pill pill-write">✍️ Write Story</div>
        <div class="arrow-sep">→</div>
        <div class="pill pill-hear">🔊 Hear It!</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["🖼️ Upload a Picture", "📷 Take a Photo"])

with tab1:
    uploaded_file = st.file_uploader(
        "Drop your favourite picture here! (JPG · PNG · WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

with tab2:
    camera_file = st.camera_input("Point your camera and snap! 📸")

# Whichever tab the user used, treat it the same way downstream
uploaded_file = uploaded_file or camera_file

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if uploaded_file is not None:

    # Save locally so Hugging Face models can read from disk
    file_path = uploaded_file.name
    with open(file_path, "wb") as fh:
        fh.write(uploaded_file.getvalue())

    st.image(uploaded_file, caption="🌊 Your magical picture!", use_column_width=True)
    st.divider()

    # ── Stage 1 : img2text ────────────────────────────────────────────────────
    with st.spinner("🔍 Taking a careful look at your picture…"):
        caption = img2text(file_path)

    if contains_taboo(caption):
        st.markdown(
            """
            <div class="safety-box">
                🛡️ The picture had some grown-up content, so we swapped it for a
                friendlier scene to keep things fun and safe for everyone!
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="card card-caption">
            <span class="card-icon">📸</span>
            <span class="card-label">What I see in your picture</span>
            <div class="card-body">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stage 2 : text2story ──────────────────────────────────────────────────
    with st.spinner("✍️ Writing a magical story just for you…"):
        story, used_fallback = text2story(caption)

    word_count = len(story.split())

    if used_fallback:
        st.markdown(
            """
            <div class="safety-box">
                🛡️ Our safety helpers spotted something not right for little ones, so
                we wrote you a brand-new adventure story instead. Enjoy!
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="card card-story">
            <span class="card-icon">📖</span>
            <span class="card-label">
                Your Story
                <span class="word-badge">{word_count} words</span>
            </span>
            <div class="card-body">{story}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stage 3 : text2audio ──────────────────────────────────────────────────
    with st.spinner("🎙️ Recording the story so you can listen along…"):
        audio_result = text2audio(story)
        wav_bytes = audio_to_wav_bytes(
            audio_result["audio"],
            audio_result["sampling_rate"],
        )

    st.markdown(
        """
        <div class="card card-audio">
            <span class="card-icon">🔊</span>
            <span class="card-label">Listen to your story!</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── OUTPUT (Properly Indented!) ───────────────────────────────────────────
    st.audio(io.BytesIO(wav_bytes), format="audio/wav")
    
    st.download_button(
        label="⬇️ Download My Story Audio",
        data=wav_bytes,
        file_name="my_story.wav",
        mime="audio/wav",
    )
    
    st.success("🎉 All done! Your story is ready — enjoy reading and listening! 🌊")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="footer">
        Made with ❤️ for little storytellers &nbsp;·&nbsp;
        ISOM5240 Individual Assignment &nbsp;·&nbsp; Powered by Hugging Face 🤗
    </div>
    """,
    unsafe_allow_html=True,
)

```
