import streamlit as st
import whisper
import os
import tempfile
# sounddevice depends on the PortAudio native library which is not available
# in many cloud deployments (Streamlit Cloud, Spaces). Import it conditionally
# and fall back to upload-only UI when it's missing.
has_sounddevice = False
try:
    import sounddevice as sd
    has_sounddevice = True
except Exception:
    sd = None
    has_sounddevice = False
import soundfile as sf
import numpy as np
import shutil
from queue import Queue
import glob
import subprocess
from pathlib import Path
import traceback
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Speech-to-Text with Whisper",
    page_icon="🎤",
    layout="wide"
)

def _find_ffmpeg_executable():
    """Try multiple strategies to locate ffmpeg on Windows/macOS/Linux.

    Returns the full path to ffmpeg.exe (or ffmpeg) or None if not found.
    """
    # 1) shutil.which (respects PATH)
    p = shutil.which("ffmpeg")
    if p:
        return p

    # 2) try the Windows `where` command (may find winget-installed path)
    try:
        res = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout:
            first = res.stdout.splitlines()[0].strip()
            if first and Path(first).exists():
                return first
    except Exception:
        pass

    # 3) search common install locations (WinGet, Program Files, tools)
    home = Path.home()
    candidates = []
    # winget package layout
    candidates += glob.glob(str(home / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Packages' / 'Gyan.FFmpeg_*' / '*' / 'bin' / 'ffmpeg.exe'))
    # common program locations
    candidates += [r'C:\Program Files\ffmpeg\bin\ffmpeg.exe', r'C:\tools\ffmpeg\bin\ffmpeg.exe']
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c))

    # 4) last resort: search PATH-like locations (may be slow)
    try:
        for root in [home, Path('C:/')]:
            for p in root.rglob('ffmpeg.exe'):
                return str(p)
    except Exception:
        pass

    return None

# Ensure ffmpeg is discoverable by the running process. If the process started
# before PATH was updated (common after winget/choco installs), try to locate
# ffmpeg using multiple heuristics and set FFMPEG_BINARY and PATH accordingly.
_ffmpeg_path = _find_ffmpeg_executable()
if _ffmpeg_path:
    os.environ.setdefault("FFMPEG_BINARY", _ffmpeg_path)
    ff_dir = str(Path(_ffmpeg_path).parent)
    if ff_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = ff_dir + os.pathsep + os.environ.get('PATH', '')


# Title and description
st.title("🎤 Speech-to-Text Application")
st.markdown("Convert audio to text using OpenAI's Whisper model - Works 100% locally!")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_size = st.selectbox(
        "Select Whisper Model",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,  # Default to 'base'
        help="""
        - tiny: Fastest, least accurate (~1GB RAM)
        - base: Fast, good for most uses (~1GB RAM)
        - small: Balanced speed/accuracy (~2GB RAM)
        - medium: High accuracy, slower (~5GB RAM)
        - large: Best accuracy, slowest (~10GB RAM)
        """
    )
    
    st.info(f"**Selected Model:** {model_size}")
    
    # Language selection (optional)
    language = st.selectbox(
        "Language (optional)",
        options=["auto"] + ["en", "es", "fr", "de", "it", "pt", "nl", "hi", "ja", "ko", "zh"],
        help="Select 'auto' for automatic detection"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses OpenAI Whisper for speech recognition.
    - No internet required
    - No API keys needed
    - Runs completely locally
    """)

# Initialize session state for model caching
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.current_model_size = None

# Function to load Whisper model
@st.cache_resource
def load_model(model_name):
    """Load and cache the Whisper model"""
    with st.spinner(f"Loading {model_name} model... This may take a minute on first run."):
        model = whisper.load_model(model_name)
    return model

# Function to transcribe audio
def transcribe_audio(audio_path, model, language="auto"):
    """Transcribe audio file using Whisper"""
    try:
        # Try to locate ffmpeg and ensure its directory is on PATH for subprocess-based calls
        ffmpeg_exec = shutil.which("ffmpeg")
        # If ffmpeg was found, ensure its directory is on PATH for subprocess-based calls
        if ffmpeg_exec:
            ffmpeg_dir = os.path.dirname(ffmpeg_exec)
            if ffmpeg_dir not in os.environ.get('PATH',''):
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH','')

        # Transcribe with progress indication
        if language == "auto":
            result = model.transcribe(audio_path)
        else:
            result = model.transcribe(audio_path, language=language)
        
        return result["text"], result.get("language", "unknown")
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        # show traceback to help debugging
        st.text_area("Traceback", traceback.format_exc(), height=300)
        return None, None

# Function to record audio
def record_audio(duration, sample_rate=16000):
    """Record audio from microphone"""
    st.info(f"🔴 Recording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished
    
    return recording, sample_rate

# Main application layout
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])

# Tab 1: Upload Audio File
with tab1:
    st.subheader("Upload an Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # Transcribe button
        if st.button("🎯 Transcribe Audio", key="transcribe_upload"):
            # Load model if not already loaded or if model size changed
            if st.session_state.model is None or st.session_state.current_model_size != model_size:
                st.session_state.model = load_model(model_size)
                st.session_state.current_model_size = model_size
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Transcribing audio... Please wait.")
                progress_bar.progress(50)
                
                # Transcribe
                start_time = time.time()
                transcription, detected_lang = transcribe_audio(
                    tmp_file_path,
                    st.session_state.model,
                    language if language != "auto" else None
                )
                end_time = time.time()
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if transcription:
                    # Display results
                    st.success(f"✅ Transcription completed in {end_time - start_time:.2f} seconds!")
                    
                    # Show detected language
                    if detected_lang:
                        st.info(f"🌐 Detected Language: {detected_lang}")
                    
                    # Display transcription
                    st.subheader("📝 Transcription:")
                    st.text_area(
                        "Transcribed Text",
                        transcription,
                        height=200,
                        label_visibility="collapsed"
                    )
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Transcription",
                        data=transcription,
                        file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

# Tab 2: Record Audio
with tab2:
    st.subheader("Record Audio from Microphone")

    # If sounddevice / PortAudio is not available (common on cloud), show
    # a friendly message and offer upload fallback. This avoids an import-time
    # OSError and allows the app to run on Streamlit Cloud / Spaces.
    if not has_sounddevice:
        st.warning(
            "Microphone recording is not available on this deployment because the PortAudio native library is missing."
        )
        st.info("You can either: (1) Upload an audio file in the Upload tab, or (2) run the app locally to use the microphone.")

        # Provide an inline uploader so users on cloud can still attach recorded files
        uploaded_record = st.file_uploader("Or upload a recorded audio file to transcribe", type=["wav","mp3","m4a","ogg","flac"]) 
        if uploaded_record is not None:
            st.audio(uploaded_record, format=f"audio/{uploaded_record.name.split('.')[-1]}")
            if st.button("🎯 Transcribe Uploaded Recording", key="transcribe_uploaded_record"):
                # Load model if needed
                if st.session_state.model is None or st.session_state.current_model_size != model_size:
                    st.session_state.model = load_model(model_size)
                    st.session_state.current_model_size = model_size

                # Save and transcribe
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_record.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_record.getvalue())
                    tmp_path = tmp_file.name
                try:
                    progress = st.progress(0)
                    progress.progress(30)
                    start_time = time.time()
                    transcription, detected_lang = transcribe_audio(tmp_path, st.session_state.model, language if language != "auto" else None)
                    end_time = time.time()
                    progress.progress(100)
                    if transcription:
                        st.success(f"✅ Transcription completed in {end_time - start_time:.2f} seconds!")
                        if detected_lang:
                            st.info(f"🌐 Detected Language: {detected_lang}")
                        st.subheader("📝 Transcription:")
                        st.text_area("Transcribed Text", transcription, height=200, label_visibility="collapsed")
                        st.download_button(label="⬇️ Download Transcription", data=transcription,
                                           file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                           mime="text/plain")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        # End of cloud fallback
    else:
        # Original server-side recording flow (sounddevice)
        # We record until the user stops. Use a Start / Stop button and store
        # recording state in session_state so the stream can run in the background.
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'rec_samplerate' not in st.session_state:
            st.session_state.rec_samplerate = 16000
        if 'rec_stream' not in st.session_state:
            st.session_state.rec_stream = None
        if 'last_record_path' not in st.session_state:
            st.session_state.last_record_path = None

        col1, col2 = st.columns([1, 1])
        with col1:
            if not st.session_state.is_recording:
                if st.button("🔴 Start Recording", key="start_record"):
                    # start recording
                    st.session_state.rec_samplerate = 16000
                    try:
                        q = Queue()

                        def _audio_callback(indata, frames, time_info, status):
                            try:
                                q.put(indata.copy(), block=False)
                            except Exception:
                                pass

                        stream = sd.InputStream(samplerate=st.session_state.rec_samplerate,
                                                 channels=1,
                                                 dtype='float32',
                                                 callback=_audio_callback)
                        stream.start()
                        st.session_state.rec_stream = stream
                        st.session_state.rec_queue = q
                        st.session_state.is_recording = True
                        st.success("Recording started. Click Stop when finished.")
                    except Exception as e:
                        st.error(f"Could not start recording: {e}")
            else:
                if st.button("⏹️ Stop Recording", key="stop_record"):
                    # stop and save
                    try:
                        stream = st.session_state.rec_stream
                        q = st.session_state.rec_queue if 'rec_queue' in st.session_state else None
                        if stream is not None:
                            stream.stop()
                            stream.close()
                    except Exception:
                        pass

                    time.sleep(0.1)

                    frames = []
                    if q is not None:
                        while not q.empty():
                            try:
                                frames.append(q.get(block=False))
                            except Exception:
                                break

                    if frames:
                        audio_np = np.concatenate(frames, axis=0)
                    else:
                        audio_np = np.empty((0,1), dtype='float32')

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        try:
                            sf.write(tmp_file.name, audio_np, st.session_state.rec_samplerate)
                            st.session_state.last_record_path = tmp_file.name
                        except Exception as e:
                            st.error(f"Failed to save recording: {e}")

                    st.session_state.is_recording = False
                    st.success("✅ Recording completed!")

        with col2:
            if st.session_state.last_record_path:
                st.audio(st.session_state.last_record_path, format="audio/wav")
                # Transcribe button
                if st.button("🎯 Transcribe Recording", key="transcribe_record"):
                    # Load model if not already loaded
                    if st.session_state.model is None or st.session_state.current_model_size != model_size:
                        st.session_state.model = load_model(model_size)
                        st.session_state.current_model_size = model_size

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Transcribing recording... Please wait.")
                    progress_bar.progress(50)

                    start_time = time.time()
                    transcription, detected_lang = transcribe_audio(
                        st.session_state.last_record_path,
                        st.session_state.model,
                        language if language != "auto" else None
                    )
                    end_time = time.time()

                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                    if transcription:
                        st.success(f"✅ Transcription completed in {end_time - start_time:.2f} seconds!")
                        if detected_lang:
                            st.info(f"🌐 Detected Language: {detected_lang}")
                        st.subheader("📝 Transcription:")
                        st.text_area("Transcribed Text", transcription, height=200, label_visibility="collapsed")
                        st.download_button(label="⬇️ Download Transcription", data=transcription,
                                           file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                           mime="text/plain")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with Streamlit & OpenAI Whisper | Runs 100% locally | No API keys required
    </div>
    """,
    unsafe_allow_html=True
)