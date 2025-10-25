import streamlit as st
import torch
import torchaudio
import whisper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, AutoModelForCTC
import librosa
import soundfile as sf
import sounddevice as sd
import numpy as np
import tempfile
import os
from datetime import datetime
import time

# =======================
# PAGE CONFIGURATION
# =======================
st.set_page_config(
    page_title="Multilingual Speech-to-Text",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# LANGUAGE CONFIGURATIONS
# =======================
LANGUAGE_MODELS = {
    # European Languages
    "English": "facebook/wav2vec2-large-960h-lv60-self",
    "Spanish": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "French": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "German": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "Italian": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "Portuguese": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "Russian": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "Dutch": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "Polish": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "Turkish": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "Greek": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    
    # Asian Languages
    "Chinese": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "Japanese": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "Korean": "kresnik/wav2vec2-large-xlsr-korean",
    "Arabic": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "Persian": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "Vietnamese": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    "Thai": "wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm",
    "Indonesian": "jonatasgrosman/wav2vec2-large-xlsr-53-indonesian",
    
    # Indian Languages (South Asian)
    "Hindi": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200",
    "Tamil": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
    "Telugu": "Harveenchadha/vakyansh-wav2vec2-telugu-tel-300",
    "Kannada": "Harveenchadha/vakyansh-wav2vec2-kannada-knr-140",
    "Malayalam": "Harveenchadha/vakyansh-wav2vec2-malayalam-mlm-100",
    "Bengali": "Harveenchadha/vakyansh-wav2vec2-bengali-bem-60",
    "Marathi": "Harveenchadha/vakyansh-wav2vec2-marathi-mrm-80",
    "Gujarati": "Harveenchadha/vakyansh-wav2vec2-gujarati-gum-70",
    "Punjabi": "Harveenchadha/vakyansh-wav2vec2-punjabi-pam-60",
    "Odia": "Harveenchadha/vakyansh-wav2vec2-odia-orm-60",
    "Urdu": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    
    # African Languages
    "Swahili": "lucio/wav2vec2-large-xlsr-swahili",
    
    # Other Languages
    "Ukrainian": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "Czech": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "Swedish": "KBLab/wav2vec2-large-voxrex-swedish",
    
    # Multilingual
    "Multilingual (Auto)": "facebook/mms-1b-all"  # Supports 1000+ languages
}

LANGUAGE_FLAGS = {
    # European Languages
    "English": "🇬🇧",
    "Spanish": "🇪🇸",
    "French": "🇫🇷",
    "German": "🇩🇪",
    "Italian": "🇮🇹",
    "Portuguese": "🇵🇹",
    "Russian": "🇷🇺",
    "Dutch": "🇳🇱",
    "Polish": "🇵🇱",
    "Turkish": "🇹🇷",
    "Greek": "🇬🇷",
    
    # Asian Languages
    "Chinese": "🇨🇳",
    "Japanese": "🇯🇵",
    "Korean": "🇰🇷",
    "Arabic": "🇸🇦",
    "Persian": "🇮🇷",
    "Vietnamese": "🇻🇳",
    "Thai": "🇹🇭",
    "Indonesian": "🇮🇩",
    
    # Indian Languages
    "Hindi": "🇮🇳",
    "Tamil": "🇮🇳",
    "Telugu": "🇮🇳",
    "Kannada": "🇮🇳",
    "Malayalam": "🇮🇳",
    "Bengali": "🇮🇳",
    "Marathi": "🇮🇳",
    "Gujarati": "🇮🇳",
    "Punjabi": "🇮🇳",
    "Odia": "🇮🇳",
    "Urdu": "🇵🇰",
    
    # African Languages
    "Swahili": "🇰🇪",
    
    # Other Languages
    "Ukrainian": "🇺🇦",
    "Czech": "🇨🇿",
    "Swedish": "🇸🇪",
    
    # Multilingual
    "Multilingual (Auto)": "🌍"
}

# =======================
# CUSTOM CSS FOR COLORFUL UI
# =======================
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C3E50 0%, #3498DB 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #4ECDC4;
        font-size: 1.1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
    }
    
    /* Language selector */
    .language-badge {
        display: inline-block;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown("""
<div class="main-header">
    <h1>🌍 Multilingual Speech-to-Text</h1>
    <p>Powered by Hugging Face Transformers | 15+ Languages | 100% Free</p>
</div>
""", unsafe_allow_html=True)

# =======================
# SIDEBAR
# =======================
with st.sidebar:
    st.markdown("### 🌐 Language Selection")
    
    # Language selector with flags
    selected_language = st.selectbox(
        "Choose Language",
        options=list(LANGUAGE_MODELS.keys()),
        format_func=lambda x: f"{LANGUAGE_FLAGS[x]} {x}",
        index=0
    )
    
    # Display selected model info
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
        <p style='color: white; margin: 0;'><b>Selected Model:</b></p>
        <p style='color: #f0f0f0; font-size: 0.85rem; margin: 0.5rem 0 0 0;'>{LANGUAGE_MODELS[selected_language]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Audio settings
    st.markdown("### 🎚️ Audio Settings")
    sample_rate = st.selectbox(
        "Sample Rate (Hz)",
        options=[16000, 22050, 44100],
        index=0,
        help="16000 Hz is optimal for speech recognition"
    )
    
    recording_duration = st.slider(
        "Recording Duration (sec)",
        min_value=3,
        max_value=30,
        value=5,
        step=1
    )
    st.markdown("---")

    # Whisper backend option
    use_whisper = st.checkbox("Use Whisper backend (more robust, slower)", value=False)
    whisper_size = "base"
    if use_whisper:
        whisper_size = st.selectbox("Whisper model size", options=["base", "small", "medium", "large"], index=0, help="Larger models are more accurate but slower and bigger on disk")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Session Stats")
    if 'transcription_count' not in st.session_state:
        st.session_state.transcription_count = 0
    if 'total_duration' not in st.session_state:
        st.session_state.total_duration = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Transcriptions", st.session_state.transcription_count)
    with col2:
        st.metric("Total Time", f"{st.session_state.total_duration:.1f}s")
    
    st.markdown("---")
    
    # Supported languages display
    st.markdown("### 🗣️ Supported Languages")
    with st.expander("View All Languages"):
        for lang, flag in LANGUAGE_FLAGS.items():
            st.markdown(f"{flag} {lang}")
    
    st.markdown("---")
    
    # About section
    st.markdown("### 💡 About")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
    <p style='color: white; margin: 0; font-size: 0.9rem;'>
    This app uses Hugging Face's fine-tuned Wav2Vec2 models for multilingual speech recognition. 
    Models are optimized for each language!
    </p>
    </div>
    """, unsafe_allow_html=True)

# =======================
# MODEL LOADING
# =======================
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    """Load Wav2Vec2 model and processor for selected language"""
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCTC.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size="base"):
    """Load OpenAI Whisper model (cached)"""
    try:
        model = whisper.load_model(model_size)
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# Load model with loading animation
model_name = LANGUAGE_MODELS[selected_language]

with st.spinner(f"🚀 Loading {selected_language} model... (First time may take 1-2 minutes)"):
    processor, model, device = load_model(model_name)

# Load Whisper model if user selected that backend
whisper_model = None
if 'use_whisper' in locals() and use_whisper:
    with st.spinner(f"🚀 Loading Whisper ({whisper_size}) model... This may take a while"):
        whisper_model = load_whisper_model(whisper_size)

if processor and model:
    st.success(f"✅ {selected_language} model loaded successfully!")
    
    # Show device info with custom styling
    device_emoji = "🖥️" if device.type == "cpu" else "🚀"
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center;'>
        <p style='color: white; margin: 0; font-size: 1.1rem;'>
            {device_emoji} Running on: <b>{device.type.upper()}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Failed to load model. Please try again or select a different language.")
    st.stop()

# =======================
# TRANSCRIPTION FUNCTION
# =======================
def transcribe_audio(audio_path, use_whisper=False, whisper_model=None):
    """Transcribe audio file using selected language model or Whisper backend

    Returns normalized transcription string or None on error.
    """
    try:
        if use_whisper and whisper_model is not None:
            # Whisper expects a path and will auto-handle resampling
            lang_code = None
            if selected_language == "English":
                lang_code = "en"
            # Run whisper transcription
            with st.spinner("🔄 Whisper is transcribing (this may be slower)"):
                if lang_code:
                    result = whisper_model.transcribe(audio_path, language=lang_code)
                else:
                    result = whisper_model.transcribe(audio_path)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            text = text.strip()
            if text:
                text = text.lower().capitalize()
            return text

        # Fallback to Wav2Vec2 transformer-based model
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        # Normalize casing to sentence case
        if transcription:
            transcription = transcription.strip()
            transcription = transcription.lower().capitalize()
        return transcription

    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")
        return None

# =======================
# RECORDING FUNCTION
# =======================
def record_audio(duration, sample_rate):
    """Record audio from microphone"""
    st.info(f"🔴 Recording for {duration} seconds... Speak now!")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Record with progress updates
    recording = []
    chunk_size = int(sample_rate * 0.1)  # 0.1 second chunks
    chunks = int(duration / 0.1)
    
    for i in range(chunks):
        chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        recording.append(chunk)
        progress_bar.progress((i + 1) / chunks)
    
    audio = np.concatenate(recording, axis=0)
    progress_bar.empty()
    
    return audio, sample_rate

# =======================
# MAIN APPLICATION
# =======================
tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio", "🌍 Language Guide"])

# ----------------
# TAB 1: Upload Audio
# ----------------
with tab1:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown(f"### 📤 Upload Your Audio File ({LANGUAGE_FLAGS[selected_language]} {selected_language})")
    st.markdown("Supported formats: **WAV, MP3, M4A, OGG, FLAC**")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        col1.metric("📝 File", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
        col2.metric("📦 Type", uploaded_file.type.split('/')[-1].upper())
        col3.metric("💾 Size", f"{uploaded_file.size / 1024:.2f} KB")
        
        st.markdown("---")
        
        # Audio player
        st.audio(uploaded_file, format=uploaded_file.type)
        
        # Transcribe button
        if st.button("🎯 Transcribe Now", key="transcribe_upload", type="primary"):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Transcribe with timing
                start_time = time.time()

                with st.spinner(f"🔄 Processing audio in {selected_language}..."):
                    transcription = transcribe_audio(tmp_file_path, use_whisper=('use_whisper' in locals() and use_whisper), whisper_model=whisper_model)

                end_time = time.time()
                duration = end_time - start_time
                
                if transcription:
                    # Update stats
                    st.session_state.transcription_count += 1
                    st.session_state.total_duration += duration
                    
                    # Success message
                    st.balloons()
                    st.success(f"✅ Transcription completed in {duration:.2f} seconds!")
                    
                    # Display language badge
                    st.markdown(f"""
                    <div class="language-badge">
                        {LANGUAGE_FLAGS[selected_language]} Transcribed in: {selected_language}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display transcription
                    st.markdown("### 📝 Transcription Result")
                    st.text_area(
                        "Your transcribed text:",
                        transcription,
                        height=200,
                        key="upload_result"
                    )
                    
                    # Word count and stats
                    word_count = len(transcription.split())
                    char_count = len(transcription)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Words", word_count)
                    col2.metric("🔤 Characters", char_count)
                    col3.metric("⚡ Speed", f"{duration:.2f}s")
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Transcription",
                        data=transcription,
                        file_name=f"transcription_{selected_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        type="primary"
                    )
            
            finally:
                # Cleanup
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------
# TAB 2: Record Audio
# ----------------
with tab2:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown(f"### 🎤 Record Audio ({LANGUAGE_FLAGS[selected_language]} {selected_language})")
    
    # Recording controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"⏱️ Recording will last **{recording_duration} seconds** in **{selected_language}**")
    
    with col2:
        record_button = st.button("🔴 Record", key="record_btn", type="primary")
    
    if record_button:
        # Record audio
        audio_data, sr = record_audio(recording_duration, sample_rate)
        
        st.success("✅ Recording completed!")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio_data, sr)
            tmp_file_path = tmp_file.name
        
        try:
            # Play recorded audio
            st.audio(tmp_file_path, format="audio/wav")
            
            # Transcribe
            with st.spinner(f"🔄 Transcribing your speech in {selected_language}..."):
                start_time = time.time()
                transcription = transcribe_audio(tmp_file_path, use_whisper=('use_whisper' in locals() and use_whisper), whisper_model=whisper_model)
                end_time = time.time()
                duration = end_time - start_time
            
            if transcription:
                # Update stats
                st.session_state.transcription_count += 1
                st.session_state.total_duration += duration
                
                # Success
                st.balloons()
                st.success(f"✅ Transcription completed in {duration:.2f} seconds!")
                
                # Display language badge
                st.markdown(f"""
                <div class="language-badge">
                    {LANGUAGE_FLAGS[selected_language]} Transcribed in: {selected_language}
                </div>
                """, unsafe_allow_html=True)
                
                # Display result
                st.markdown("### 📝 Transcription Result")
                st.text_area(
                    "Your transcribed text:",
                    transcription,
                    height=200,
                    key="record_result"
                )
                
                # Word count and stats
                word_count = len(transcription.split())
                char_count = len(transcription)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Words", word_count)
                col2.metric("🔤 Characters", char_count)
                col3.metric("⚡ Speed", f"{duration:.2f}s")
                
                # Download
                st.download_button(
                    label="⬇️ Download Transcription",
                    data=transcription,
                    file_name=f"recording_{selected_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    type="primary"
                )
        
        finally:
            # Cleanup
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------
# TAB 3: Language Guide
# ----------------
with tab3:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("### 🌍 Multilingual Support Guide")
    
    st.markdown("""
    This application supports **35+ languages** including **10 Indian languages** 
    using state-of-the-art Wav2Vec2 models fine-tuned specifically for each language!
    """)
    
    st.markdown("---")
    
    # Language features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Key Features")
        st.markdown("""
        - 🌐 **35+ Languages** supported
        - 🇮🇳 **10 Indian Languages** included
        - 🎯 **Language-specific models** for accuracy
        - 🚀 **Fast transcription** (GPU accelerated)
        - 💾 **No API keys** required
        - 🔒 **100% Local** processing
        - 📱 **Easy language switching**
        """)
    
    with col2:
        st.markdown("#### 🗣️ Supported Languages")
        st.markdown("**🇮🇳 Indian Languages:**")
        indian_langs = ["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Bengali", "Marathi", "Gujarati", "Punjabi", "Odia"]
        for lang in indian_langs:
            st.markdown(f"{LANGUAGE_FLAGS[lang]} **{lang}**")
        
        with st.expander("Show All Other Languages"):
            other_langs = [l for l in LANGUAGE_FLAGS.keys() if l not in indian_langs]
            for lang in other_langs:
                st.markdown(f"{LANGUAGE_FLAGS[lang]} **{lang}**")
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips for Best Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🎤 Audio Quality**
        - Clear audio recording
        - Minimize background noise
        - Normal speaking pace
        - Good microphone quality
        - Proper audio levels
        """)
    
    with col2:
        st.success("""
        **🌐 Language Selection**
        - Choose correct language
        - Use native speaker audio
        - Clear pronunciation
        - Avoid mixed languages
        - One language at a time
        """)
    
    with col3:
        st.warning("""
        **⚙️ Technical Tips**
        - Use 16kHz sample rate
        - Keep audio under 30s
        - GPU faster than CPU
        - First load takes time
        - Models cached locally
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Comparison")
    
    st.markdown("""
    | Language | Model Size | Accuracy | Speed |
    |----------|-----------|----------|-------|
    | English | ~1.2GB | ⭐⭐⭐⭐⭐ | Fast |
    | Hindi/Tamil/Telugu | ~1.2GB | ⭐⭐⭐⭐ | Fast |
    | Kannada/Malayalam | ~1.2GB | ⭐⭐⭐⭐ | Fast |
    | Spanish/French | ~1.2GB | ⭐⭐⭐⭐ | Fast |
    | Arabic/Chinese | ~1.2GB | ⭐⭐⭐⭐ | Fast |
    | Multilingual | ~3.8GB | ⭐⭐⭐⭐ | Medium |
    """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 How It Works")
    
    st.markdown("""
    1. **Select Language**: Choose your target language from the sidebar
    2. **Upload/Record**: Either upload an audio file or record directly
    3. **AI Processing**: Wav2Vec2 model processes the audio
    4. **Get Results**: Receive accurate transcription instantly
    5. **Download**: Save your transcription as a text file
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =======================
# FOOTER
# =======================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px;'>
    <p style='color: white; font-size: 1.2rem; margin: 0;'>
        🌍 <b>Multilingual Speech-to-Text</b>
    </p>
    <p style='color: #f0f0f0; font-size: 1rem; margin: 0.5rem 0;'>
        Currently using: {LANGUAGE_FLAGS[selected_language]} <b>{selected_language}</b>
    </p>
    <p style='color: #f0f0f0; font-size: 0.9rem; margin-top: 0.5rem;'>
        Built with ❤️ using Streamlit & Hugging Face Transformers<br>
        100% Free | No API Keys | Runs Locally | 35+ Languages | 10 Indian Languages 🇮🇳
    </p>
</div>
""", unsafe_allow_html=True)