import streamlit as st
import io
import numpy as np
import scipy.io.wavfile as wav


from transformers import pipeline

# Function part
def img2text(url: str) -> str:
    """
    Stage 1 – Image → Caption.

    Accepts a local file path (str) to an image and returns a plain-text
    caption describing its contents.

    Args:
        url: Local path to the saved image file.

    Returns:
        A short descriptive caption produced by the BLIP model.
    """
    image_to_text_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    text = image_to_text_model(url)[0]["generated_text"]
    return text


def text2story(text: str) -> str:
    """
    Stage 2 – Caption → Children's Story.

    Expands the image caption into a short, child-friendly fairy-tale
    narrative (target length: 50–100 words).

    Args:
        text: The image caption produced by img2text.

    Returns:
        A generated story string trimmed to ~100 words ending at a
        sentence boundary.
    """
    story_pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=130,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.25,
        do_sample=True,
    )

    # Fairy-tale style prompt prefix to guide the model
    prompt = (
        f"Once upon a time, {text}. "
        "It was a magical day full of wonder and excitement. "
        "Suddenly"
    )

    result = story_pipe(prompt)[0]["generated_text"]

    # ── Trim to ≤ 100 words, ending at a sentence boundary ──
    words = result.split()
    if len(words) > 100:
        trimmed = " ".join(words[:100])
        # Find the last full-stop, exclamation, or question mark
        last_stop = max(
            trimmed.rfind("."),
            trimmed.rfind("!"),
            trimmed.rfind("?"),
        )
        story_text = trimmed[: last_stop + 1] if last_stop != -1 else trimmed
    else:
        story_text = " ".join(words)

    return story_text


def text2audio(story_text: str) -> dict:
    """
    Stage 3 – Story → Audio.

    Converts the generated story into speech using the Hugging Face
    MMS English TTS model.

    Args:
        story_text: The story string to be spoken aloud.

    Returns:
        A dict containing:
            'audio'         – numpy float32 array of audio samples
            'sampling_rate' – integer sample rate in Hz
    """
    tts_pipe = pipeline(
        "text-to-speech",
        model="Matthijs/mms-tts-eng",
    )
    audio_data = tts_pipe(story_text)
    return audio_data


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """
    Helper – convert a numpy float audio array to raw WAV bytes.

    Normalises the signal to [-1, 1] and encodes as 16-bit PCM WAV,
    which Streamlit's st.audio widget can play natively.

    Args:
        audio_array:  1-D (or squeezable) numpy float array of samples.
        sample_rate:  Sample rate in Hz.

    Returns:
        WAV-encoded bytes ready for st.audio().
    """
    audio_array = np.squeeze(audio_array).astype(np.float32)

    # Normalise to [-1, 1] if the signal is outside that range
    peak = np.abs(audio_array).max()
    if peak > 0:
        audio_array = audio_array / peak

    audio_int16 = (audio_array * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wav.write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    return buffer.read()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StoryMagic ✨",
    page_icon="📖",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (kid-friendly, whimsical storybook theme)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Bubblegum+Sans&family=Nunito:wght@400;600;700;800&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        background-color: #FFF8EE;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #FFD166 0%, #FF6B6B 55%, #A78BFA 100%);
        border-radius: 28px;
        padding: 2.2rem 1.8rem 1.8rem;
        text-align: center;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 40px rgba(255,107,107,0.28);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "⭐🌙☁️🌈⭐";
        position: absolute;
        top: 0.5rem; left: 50%;
        transform: translateX(-50%);
        font-size: 1.1rem;
        letter-spacing: 1.2rem;
        opacity: 0.55;
    }
    .hero h1 {
        font-family: 'Bubblegum Sans', cursive;
        font-size: 3rem;
        color: #fff;
        text-shadow: 3px 3px 0 rgba(0,0,0,0.14);
        margin: 0.6rem 0 0.4rem;
    }
    .hero p {
        color: rgba(255,255,255,0.93);
        font-size: 1.08rem;
        font-weight: 700;
        margin: 0;
    }

    /* ── Step pipeline row ── */
    .steps-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.8rem;
    }
    .step-pill {
        background: #fff;
        border: 2.5px solid #FFD166;
        border-radius: 50px;
        padding: 0.35rem 1rem;
        font-size: 0.83rem;
        font-weight: 800;
        color: #666;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        white-space: nowrap;
    }
    .arrow { font-size: 1.2rem; color: #ccc; }

    /* ── Result cards ── */
    .card {
        background: #fff;
        border-radius: 22px;
        padding: 1.4rem 1.7rem;
        margin: 1rem 0;
        box-shadow: 0 4px 22px rgba(0,0,0,0.07);
        border-left: 7px solid #FFD166;
        animation: fadeUp 0.45s ease both;
    }
    .card.story  { border-left-color: #FF6B6B; }
    .card.audio  { border-left-color: #6BCB77; }
    .card-label {
        font-family: 'Bubblegum Sans', cursive;
        font-size: 1rem;
        color: #aaa;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .card-body {
        font-size: 1.08rem;
        color: #333;
        line-height: 1.75;
    }

    /* ── File uploader styling ── */
    [data-testid="stFileUploader"] > div > div {
        border: 3px dashed #FFD166 !important;
        border-radius: 20px !important;
        background: #FFFBF2 !important;
    }

    /* ── Success message ── */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        font-weight: 700;
    }

    /* ── Animations ── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #ccc;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1.5px dashed #eee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>📖 StoryMagic ✨</h1>
        <p>Upload any picture and watch your very own fairy tale come to life! 🎧🌟</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline step badges ──────────────────────────────────────────────────────
st.markdown(
    """
    <div class="steps-row">
        <div class="step-pill">📸 Upload Image</div>
        <div class="arrow">→</div>
        <div class="step-pill">🔍 Read Image</div>
        <div class="arrow">→</div>
        <div class="step-pill">✍️ Write Story</div>
        <div class="arrow">→</div>
        <div class="step-pill">🔊 Hear It!</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main part


uploaded_file = st.file_uploader(
    "🖼️ Drop your favourite picture here (JPG · PNG · WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE  (runs only when an image has been uploaded)
# ──────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:

    # ── Save the uploaded file locally so Hugging Face can read it ──
    file_path = uploaded_file.name
    with open(file_path, "wb") as fh:
        fh.write(uploaded_file.getvalue())

    # Display the uploaded image
    st.image(
        uploaded_file,
        caption="🌟 Your magical picture",
        use_column_width=True,
    )
    st.divider()

    # ── Stage 1 : Image → Caption (img2text) ─────────────────────────────────
    with st.spinner("🔍 Taking a close look at your picture…"):
        caption = img2text(file_path)

    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">📸 What I see</div>
            <div class="card-body">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stage 2 : Caption → Story (text2story) ───────────────────────────────
    with st.spinner("✍️ Writing a magical story just for you…"):
        story = text2story(caption)

    st.markdown(
        f"""
        <div class="card story">
            <div class="card-label">📖 Your Story</div>
            <div class="card-body">{story}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stage 3 : Story → Audio (text2audio) ─────────────────────────────────
    with st.spinner("🎙️ Recording the story so you can listen…"):
        audio_result = text2audio(story)
        wav_bytes = audio_to_wav_bytes(
            audio_result["audio"],
            audio_result["sampling_rate"],
        )

    st.markdown(
        """
        <div class="card audio">
            <div class="card-label">🔊 Listen to your story!</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.audio(wav_bytes, format="audio/wav")

    st.success("🎉 Woohoo! Your story is ready — enjoy reading and listening!")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="footer">
        Made with ❤️ for little storytellers &nbsp;·&nbsp; ISOM5240 Individual Assignment
    </div>
    """,
    unsafe_allow_html=True,
)

        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
