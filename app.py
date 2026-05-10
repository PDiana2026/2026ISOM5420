"""
ISOM5240 Individual Assignment
Storytelling Application using Hugging Face Pipelines
------------------------------------------------------
Pipeline:
  img2text   →  Salesforce/blip-image-captioning-base
  text2story →  roneneldan/TinyStories-33M   (trained on children's stories)
  text2audio →  Matthijs/mms-tts-eng

Safety:
  • Taboo-word list guards both caption prompt and generated story output.
  • Story is strictly trimmed / padded to 50-100 words.
  • Up to 3 regeneration attempts before a guaranteed safe fallback is used.
"""

import io
import re
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
from transformers import pipeline


# ══════════════════════════════════════════════════════════════════════════════
# CHILD-SAFETY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TABOO_WORDS: set = {
    # Violence / harm
    "murder", "kill", "killed", "killing", "kills", "killer",
    "death", "dead", "die", "dying", "dies", "died","orgasm",
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

SAFE_FALLBACK_STORY: str = (
    "Once upon a time, a cheerful little bunny named Biscuit lived near a sunny meadow. "
    "Every morning Biscuit hopped through rainbow-coloured flowers, greeting butterflies "
    "and bumble bees. One day he discovered a sparkling pond full of friendly frogs. "
    "They jumped and splashed and sang merry songs all afternoon. When the golden sun "
    "began to set, Biscuit hurried home, where his mum had warm carrot soup waiting. "
    "He snuggled into his cosy bed, dreaming of tomorrow's adventures. The End."
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — TABOO CHECKER
# ══════════════════════════════════════════════════════════════════════════════

def contains_taboo(text: str) -> bool:
    """
    Return True if *text* contains any word from TABOO_WORDS.
    Uses whole-word, case-insensitive regex matching so 'killed' won't
    accidentally block unrelated words sharing the same letters.
    """
    lowered = text.lower()
    for word in TABOO_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — WORD-COUNT ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════════

_PADDING = (
    " Together they laughed and played until the golden sun dipped below the hills. "
    "Everyone went home happy. The End."
)


def enforce_word_count(text: str) -> str:
    """
    Guarantee the returned story is between MIN_WORDS and MAX_WORDS words.

    - Too long  → trim at the last sentence boundary inside the word limit.
    - Too short → append a cheerful closing sentence, then re-trim if needed.
    """
    def _trim(s: str) -> str:
        words = s.split()
        if len(words) <= MAX_WORDS:
            return s
        candidate = " ".join(words[:MAX_WORDS])
        last = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
        return candidate[: last + 1] if last != -1 else candidate

    text = _trim(text)

    # Pad if still too short after trimming
    if len(text.split()) < MIN_WORDS:
        text = _trim(text.rstrip() + _PADDING)

    return text


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — IMAGE → CAPTION
# ══════════════════════════════════════════════════════════════════════════════

def img2text(url: str) -> str:
    """
    Convert an uploaded image to a plain-text description.

    Model : Salesforce/blip-image-captioning-base
    Args  : url – local file path to the saved image.
    Returns a short descriptive caption string.
    """
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    caption = captioner(url)[0]["generated_text"]
    return caption


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CAPTION → CHILDREN'S STORY (50-100 words, child-safe)
# ══════════════════════════════════════════════════════════════════════════════

def text2story(caption: str) -> tuple:
    """
    Expand a caption into a child-friendly story of 50-100 words.

    Model : roneneldan/TinyStories-33M
        Trained exclusively on synthetic short stories written for children
        aged 3-4, ensuring simple vocabulary and gentle, positive themes.

    Safety workflow
    ---------------
    1. If the caption contains taboo words, substitute a safe generic prompt.
    2. Generate a story and enforce the 50-100 word limit.
    3. Check the story for taboo content; if clean, return it.
    4. Repeat up to MAX_REGEN_ATTEMPTS times.
    5. If all attempts fail the safety check, return SAFE_FALLBACK_STORY.

    Args    : caption – image caption from img2text.
    Returns : (story_text: str, used_fallback: bool)
    """
    story_pipe = pipeline(
        "text-generation",
        model="roneneldan/TinyStories-33M",
        max_new_tokens=160,
        temperature=0.75,
        top_p=0.92,
        repetition_penalty=1.3,
        do_sample=True,
    )

    # Sanitise the prompt if the caption contains anything inappropriate
    safe_caption = (
        caption
        if not contains_taboo(caption)
        else "a friendly little animal playing in a sunny meadow"
    )

    prompt = f"Once upon a time, {safe_caption}. One happy morning,"

    for _ in range(MAX_REGEN_ATTEMPTS):
        raw = story_pipe(prompt)[0]["generated_text"]
        story = enforce_word_count(raw)
        if not contains_taboo(story):
            return story, False  # clean story found

    # All attempts failed the safety check; return guaranteed-safe fallback
    return SAFE_FALLBACK_STORY, True


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — STORY → AUDIO
# ══════════════════════════════════════════════════════════════════════════════

def text2audio(story_text: str) -> dict:
    """
    Convert the generated story to spoken audio.

    Model : Matthijs/mms-tts-eng  (Facebook MMS English TTS)
    Args  : story_text – the story string to synthesise.
    Returns a dict with keys 'audio' (numpy float32 array) and 'sampling_rate'.
    """
    tts_pipe = pipeline(
        "text-to-speech",
        model="Matthijs/mms-tts-eng",
    )
    return tts_pipe(story_text)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — NUMPY AUDIO → WAV BYTES
# ══════════════════════════════════════════════════════════════════════════════

def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode a numpy float audio array as 16-bit PCM WAV bytes for st.audio().
    Normalises the signal to [-1, 1] to avoid clipping.
    """
    arr = np.squeeze(audio_array).astype(np.float32)
    peak = np.abs(arr).max()
    if peak > 0:
        arr /= peak
    buf = io.BytesIO()
    wav.write(buf, sample_rate, (arr * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="StoryMagic ✨",
    page_icon="🌈",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  —  Toybox Maximalist children's theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@600;700;800&display=swap');

    :root {
        --sky:    #C8EDFF;
        --sun:    #FFD60A;
        --coral:  #FF4757;
        --green:  #2ED573;
        --purple: #C77DFF;
        --navy:   #1A1A2E;
        --white:  #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
        background: var(--sky) !important;
    }

    /* Polka-dot patterned background */
    .main > div {
        background:
            radial-gradient(circle, rgba(255,255,255,0.5) 2px, transparent 2px) 0 0 / 36px 36px,
            linear-gradient(160deg, #C8EDFF 0%, #D4F5E9 50%, #FFE8F5 100%);
        min-height: 100vh;
        padding-bottom: 3rem;
    }

    /* ── Keyframe animations ── */
    @keyframes floatA {
        0%,100% { transform: translateY(0)    rotate(0deg);  }
        50%      { transform: translateY(-18px) rotate(8deg); }
    }
    @keyframes floatB {
        0%,100% { transform: translateY(0)    rotate(0deg);   }
        50%      { transform: translateY(-12px) rotate(-6deg); }
    }
    @keyframes popIn {
        0%   { opacity:0; transform: scale(0.6) translateY(28px); }
        70%  { transform: scale(1.05) translateY(-4px); }
        100% { opacity:1; transform: scale(1)   translateY(0); }
    }
    @keyframes rainbowSlide {
        0%   { background-position: 0%   50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0%   50%; }
    }
    @keyframes bounce {
        0%,100% { transform: translateY(0);    }
        50%      { transform: translateY(-9px); }
    }

    /* ── Floating decoration row ── */
    .hero-deco {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        font-size: 2.1rem;
        margin-bottom: -0.9rem;
        position: relative;
        z-index: 2;
    }
    .hero-deco span:nth-child(1) { animation: floatA 3.2s ease-in-out infinite; }
    .hero-deco span:nth-child(2) { animation: floatB 2.8s ease-in-out infinite 0.4s; }
    .hero-deco span:nth-child(3) { animation: floatA 3.5s ease-in-out infinite 0.8s; }
    .hero-deco span:nth-child(4) { animation: floatB 3.0s ease-in-out infinite 1.2s; }
    .hero-deco span:nth-child(5) { animation: floatA 2.6s ease-in-out infinite 0.2s; }

    /* ── Hero banner ── */
    .hero-box {
        background: linear-gradient(270deg,
            #FF6B6B, #FFD60A, #2ED573, #48CAE4, #C77DFF, #FF6B6B);
        background-size: 400% 400%;
        animation: rainbowSlide 6s ease infinite;
        border-radius: 36px;
        border: 5px solid var(--navy);
        padding: 2rem 1.5rem 1.7rem;
        text-align: center;
        box-shadow: 8px 8px 0 var(--navy);
        margin-bottom: 0;
        position: relative;
        overflow: hidden;
    }
    .hero-box::after {
        content: "⭐  ☁️  🌙  ☁️  ⭐";
        position: absolute;
        bottom: 0.45rem; left: 50%;
        transform: translateX(-50%);
        font-size: 0.9rem;
        opacity: 0.45;
        letter-spacing: 1rem;
        white-space: nowrap;
    }
    .hero-title {
        font-family: 'Fredoka One', cursive !important;
        font-size: 3.2rem !important;
        color: var(--white) !important;
        text-shadow: 4px 4px 0 var(--navy);
        margin: 0 0 0.4rem !important;
        line-height: 1.1 !important;
    }
    .hero-sub {
        font-size: 1.08rem;
        font-weight: 800;
        color: rgba(255,255,255,0.95);
        text-shadow: 1px 1px 0 rgba(0,0,0,0.22);
        margin: 0 !important;
    }

    /* ── Step-pill row ── */
    .steps-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 1.5rem 0 2rem;
    }
    .pill {
        border: 3px solid var(--navy);
        border-radius: 50px;
        padding: 0.4rem 1.1rem;
        font-size: 0.83rem;
        font-weight: 800;
        color: var(--navy);
        box-shadow: 3px 3px 0 var(--navy);
        white-space: nowrap;
        animation: bounce 2.4s ease-in-out infinite;
    }
    .pill-1 { background: #FFD60A; animation-delay: 0.0s; }
    .pill-2 { background: #FF9FF3; animation-delay: 0.3s; }
    .pill-3 { background: #A9FF68; animation-delay: 0.6s; }
    .pill-4 { background: #74C0FC; animation-delay: 0.9s; }
    .arrow-sep { font-size: 1.4rem; color: var(--navy); font-weight: 900; }

    /* ── Result cards ── */
    .card {
        border-radius: 28px;
        border: 4px solid var(--navy);
        padding: 1.5rem 1.8rem;
        margin: 1.1rem 0;
        box-shadow: 6px 6px 0 var(--navy);
        animation: popIn 0.55s cubic-bezier(0.34,1.56,0.64,1) both;
    }
    .card-caption { background: #FFF9C4; }
    .card-story   { background: #FFE4E1; }
    .card-audio   { background: #D4EDDA; }

    .card-icon {
        font-size: 2.3rem;
        display: inline-block;
        animation: floatA 2.8s ease-in-out infinite;
        margin-bottom: 0.25rem;
    }
    .card-label {
        font-family: 'Fredoka One', cursive;
        font-size: 1.2rem;
        color: var(--navy);
        margin-bottom: 0.55rem;
        display: block;
    }
    .card-body {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2C2C54;
        line-height: 1.8;
    }
    .word-badge {
        display: inline-block;
        background: var(--navy);
        color: var(--white);
        font-size: 0.72rem;
        font-weight: 800;
        border-radius: 50px;
        padding: 0.15rem 0.7rem;
        margin-left: 0.6rem;
        vertical-align: middle;
    }

    /* ── Safety notice box ── */
    .safety-box {
        background: #FFF3CD;
        border: 3px solid #FFC107;
        border-radius: 22px;
        padding: 0.9rem 1.3rem;
        font-size: 0.92rem;
        font-weight: 700;
        color: #856404;
        box-shadow: 4px 4px 0 #E6AC00;
        margin: 0.7rem 0;
        animation: popIn 0.45s ease both;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] > div > div {
        border: 4px dashed #C77DFF !important;
        border-radius: 24px !important;
        background: rgba(199,125,255,0.08) !important;
    }

    /* ── Uploaded image frame ── */
    [data-testid="stImage"] img {
        border-radius: 24px !important;
        border: 4px solid var(--navy) !important;
        box-shadow: 8px 8px 0 var(--navy) !important;
    }

    /* ── Audio player ── */
    audio {
        width: 100%;
        border-radius: 50px;
        outline: 3px solid var(--navy);
    }

    hr { border: 2px dashed #CCC !important; border-radius: 4px; }

    [data-testid="stAlert"] {
        border-radius: 20px !important;
        border: 3px solid var(--navy) !important;
        font-weight: 800 !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 3rem;
        padding-top: 1.1rem;
        border-top: 2px dashed #DDD;
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
    <div class="hero-deco">
        <span>🌈</span><span>⭐</span><span>🦄</span><span>⭐</span><span>🎨</span>
    </div>
    <div class="hero-box">
        <div class="hero-title">✨ StoryMagic ✨</div>
        <p class="hero-sub">
            Upload any picture &amp; watch a fairy tale appear — then hear it read aloud! 🎧📖
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="steps-row">
        <div class="pill pill-1">📸 Upload Picture</div>
        <div class="arrow-sep">→</div>
        <div class="pill pill-2">🔍 Read Image</div>
        <div class="arrow-sep">→</div>
        <div class="pill pill-3">✍️ Write Story</div>
        <div class="arrow-sep">→</div>
        <div class="pill pill-4">🔊 Hear It!</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "🖼️ Drop your favourite picture here! (JPG · PNG · WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if uploaded_file is not None:

    # Persist to disk so the BLIP model can open it
    file_path = uploaded_file.name
    with open(file_path, "wb") as fh:
        fh.write(uploaded_file.getvalue())

    st.image(uploaded_file, caption="🌟 Your magical picture!", use_column_width=True)
    st.divider()

    # ── Stage 1 : img2text ────────────────────────────────────────────────────
    with st.spinner("🔍 Taking a careful look at your picture…"):
        caption = img2text(file_path)

    if contains_taboo(caption):
        st.markdown(
            """<div class="safety-box">
                🛡️ The picture had some grown-up content, so we swapped it for a
                friendlier scene to keep things fun and safe for everyone!
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""<div class="card card-caption">
            <span class="card-icon">📸</span>
            <span class="card-label">What I see in your picture</span>
            <div class="card-body">{caption}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Stage 2 : text2story ──────────────────────────────────────────────────
    with st.spinner("✍️ Writing a magical story just for you…"):
        story, used_fallback = text2story(caption)

    word_count = len(story.split())

    if used_fallback:
        st.markdown(
            """<div class="safety-box">
                🛡️ Our safety helpers spotted something not right for little ones, so
                we wrote you a brand-new adventure story instead. Enjoy!
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""<div class="card card-story">
            <span class="card-icon">📖</span>
            <span class="card-label">
                Your Story
                <span class="word-badge">{word_count} words</span>
            </span>
            <div class="card-body">{story}</div>
        </div>""",
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
        """<div class="card card-audio">
            <span class="card-icon">🔊</span>
            <span class="card-label">Listen to your story!</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.audio(wav_bytes, format="audio/wav")
    st.success("🎉 All done! Your story is ready — enjoy reading and listening! 🌈")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """<div class="footer">
        Made with ❤️ for little storytellers &nbsp;·&nbsp;
        ISOM5240 Individual Assignment &nbsp;·&nbsp; Powered by Hugging Face 🤗
    </div>""",
    unsafe_allow_html=True,
)
