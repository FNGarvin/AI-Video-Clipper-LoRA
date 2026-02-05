# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner (v5.0 Staging)
# 🏆 CREDITS: Cyberbol (Logic), FNGarvin (Engine), WildSpeaker (5090 Fix)
# --------------------------------------------------------------------------------

import os
import sys

# --- 1. BOOTSTRAP & PATCHES ---
# 🚨 CRITICAL: Patches must apply BEFORE heavy imports (Torch, WhisperX)
try:
    from modules import patches
    patches.apply_patches()
except ImportError:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Basic fallbacks if patches.py is missing
    pass

import streamlit as st
import whisperx
from moviepy import VideoFileClip, AudioFileClip
import tempfile
import torch
import gc
import time
import tkinter as tk
from tkinter import filedialog
import logging

# Suppress Streamlit's "missing ScriptRunContext" warning in threads
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# 5090 Fix (In-line backup if patches failed or specifically for this file)
if not hasattr(torch, "_patched_for_5090"):
    _orig_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _safe_load
    torch._patched_for_5090 = True

# --- EXPORT MODULES TO PATH ---
# Ensure modules can be imported if running from a subdir (future proofing)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.vision_engine import VisionEngine, scan_local_gguf_models
from modules.audio_engine import AudioEngine
from modules.downloader import download_model

# --- 2. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR

# --- 3. UI CONFIG ---
st.set_page_config(page_title="AI Clipper v5.0", layout="wide")
st.title("👁️🐧👂 AI Video Clipper & LoRA Captioner")
st.markdown("v5.0 Staging | **Cyberbol** (Logic) | **FNGarvin** (Engine) | **WildSpeaker** (Fixes)")

device = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.header("⚙️ Engine Status")
if device == "cuda":
    st.sidebar.success(f"GPU: **{torch.cuda.get_device_name(0)}**")
else:
    st.sidebar.error("CUDA not detected!")

st.sidebar.divider()
app_mode = st.sidebar.selectbox("Choose Mode:", [
    "🎥 Video Auto-Clipper", 
    "📝 Bulk Video Captioner",
    "🖼️ Image Folder Captioner"
])

# --- MODEL SELECTION (Radio Buttons) ---
model_options = {
    "GGUF: Gemma-3-12B (Next-Gen, 4-bit)": {
        "backend": "gguf",
        "repo": "unsloth/gemma-3-12b-it-GGUF",
        "model": "gemma-3-12b-it-IQ4_XS.gguf",
        "projector": "mmproj-F16.gguf"
    },
    "GGUF: Qwen3-VL-8B-Instruct (Q4_K_M)": {
        "backend": "gguf",
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "model": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "projector": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
    },
    "Transformer: Qwen2-VL-7B (High Quality)": {
        "backend": "transformers",
        "id": "Qwen/Qwen2-VL-7B-Instruct"
    },
    "Transformer: Qwen2-VL-2B (Fast)": {
        "backend": "transformers",
        "id": "Qwen/Qwen2-VL-2B-Instruct"
    }
}

# Auto-Discovery
local_ggufs = scan_local_gguf_models(MODELS_DIR)
# Filter duplicates
existing_ggufs = [m["model"] for m in model_options.values() if m.get("backend") == "gguf"]
for label, config in local_ggufs.items():
    if config["model"] not in existing_ggufs:
        model_options[label] = config

model_label = st.sidebar.radio("Vision Model:", list(model_options.keys()), index=2)
SELECTED_MODEL = model_options[model_label]
SELECTED_VISION_ID = SELECTED_MODEL.get("id", SELECTED_MODEL.get("model")) # Fallback ID

# Audio Model ID (Fixed for now, can be modularized later)
SELECTED_AUDIO_ID = "Qwen/Qwen2-Audio-7B-Instruct"

st.sidebar.divider()
st.sidebar.markdown("### 📝 Instructions")

# PROMPTY
default_prompt = "Describe this {type} in detail for a dataset. Main subject: {trigger}. Describe the action, camera movement, lighting, atmosphere, and background."
user_instruction = st.sidebar.text_area("Vision Prompt:", value=default_prompt, height=150)
lora_trigger = st.text_input("LoRA Trigger Word (Optional)", value="cbrl man")

audio_prompt_default = "Describe the sound atmosphere and music mood in one short, non-technical sentence. Do not mention BPM or keys."
audio_prompt = st.sidebar.text_area("Audio Prompt (Qwen2-Audio):", value=audio_prompt_default, height=60)

# --- SIDEBAR: ADVANCED OPTIONS ---
with st.sidebar.expander("🛠️ Advanced Generation Options"):
    gen_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    gen_top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    gen_max_tokens = st.number_input("Max New Tokens", 64, 2048, 256)
    
GEN_CONFIG = {
    "temperature": gen_temp,
    "top_p": gen_top_p,
    "max_tokens": gen_max_tokens
}

# --- 4. HELPERS ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

# --- 5. MODEL LOADERS (MODULARIZED) ---
def load_vision_engine():
    """Unified loading using VisionEngine class"""
    # Check if model changed
    if 'vision_engine' not in st.session_state or st.session_state.get('last_model_config') != str(SELECTED_MODEL):
        
        # Clear old
        if 'vision_engine' in st.session_state and st.session_state['vision_engine']:
            st.session_state['vision_engine'].clear()
            
        with st.status(f"🚀 Initializing Vision Engine ({model_label})...", expanded=True) as status:
            engine = VisionEngine(SELECTED_MODEL, device=device, models_dir=MODELS_DIR)
            engine.load(log_callback=status.write)
            status.update(label="✅ Vision Engine Ready!", state="complete", expanded=False)
            
        st.session_state['vision_engine'] = engine
        st.session_state['last_model_config'] = str(SELECTED_MODEL)
    
    return st.session_state['vision_engine']

def load_audio_engine():
    """Unified loading using AudioEngine class"""
    if 'audio_engine' not in st.session_state:
        st.session_state['audio_engine'] = None

    # Lazy load or check existence? 
    # For now, let's just return the instance, create if needed
    if not st.session_state['audio_engine']:
        engine = AudioEngine(SELECTED_AUDIO_ID, device=device, models_dir=MODELS_DIR)
        # We don't auto-load here to save VRAM, load on demand in the loop
        st.session_state['audio_engine'] = engine
        
    return st.session_state['audio_engine']

def select_folder_dialog():
    root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root); root.destroy()
    return folder_path

# =================================================================================================
# MODE 1: VIDEO AUTO-CLIPPER (v5.0 Merged Logic)
# =================================================================================================
if app_mode == "🎥 Video Auto-Clipper":
    project_name = st.text_input("Project Name (Optional)", value="")
    uploaded_file = st.file_uploader("Upload Video (MP4, MKV)", type=["mp4", "mkv"])
    
    st.subheader("✂️ Cutting Parameters")
    keep_orig = st.checkbox("Keep Original Resolution & FPS", value=False)
    col1, col2, col3, col4 = st.columns(4)
    with col1: target_dur = st.number_input("Target Length (s)", 1.0, 60.0, 5.0)
    with col2: out_width = st.number_input("Output Width", 256, 3840, 1024, disabled=keep_orig)
    with col3: out_height = st.number_input("Output Height", 256, 3840, 1024, disabled=keep_orig)
    with col4: out_fps = st.number_input("Output FPS", 1, 120, 24, disabled=keep_orig)
    
    st.markdown("---")
    
    c_ltx, c_audio = st.columns(2)
    with c_ltx:
        enable_hard_cut = st.checkbox("⚡ LTX Hard Cut Mode (Fixed Duration)", value=False, help="Strict cut. Ignores sentence end.")
        if enable_hard_cut:
            max_clips = st.number_input("Max Clips Limit", 1, 500, 20)
            tol_minus, tol_plus = 0, 0
        else:
            tol_minus = st.number_input("Tolerance - (s)", 0.0, 5.0, 0.0)
            tol_plus = st.number_input("Tolerance + (s)", 0.0, 10.0, 0.5)
            
    with c_audio:
        enable_audio_cap = st.checkbox("🎧 Enable Audio Analysis (Qwen2-Audio)", value=False, help="Adds description of background sounds.")

    col_btn, col_timer = st.columns([1, 4])
    with col_btn:
        start_processing = st.button("🚀 START PROCESSING")
    with col_timer:
        timer_placeholder = st.empty()

    if uploaded_file and start_processing:
        start_ts = time.time()
        timer_placeholder.info("⏱️ Processing started...")
        status_box = st.empty()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read()); video_path = tmp.name
        
        try:
            # === PHASE 1: WHISPER X ===
            status_box.info("🚀 **Phase 1/3: Speech Analysis (WhisperX)**...")
            
             # Use downloader to ensure model is present and get local path
            wx_repo = "Systran/faster-whisper-large-v3"
            wx_path = os.path.normpath(os.path.join(MODELS_DIR, wx_repo))
            
            # Robust Check
            essential_wx = ["config.json", "model.bin", "tokenizer.json", "vocabulary.json"]
            if not all(os.path.exists(os.path.join(wx_path, f)) for f in essential_wx):
                with st.spinner("Ensuring Whisper Model (Downloader Active)..."):
                    wx_path = download_model(wx_repo, MODELS_DIR, specific_files=essential_wx, log_callback=status_box.text)

            check_clip = VideoFileClip(video_path); video_duration = check_clip.duration; check_clip.close(); del check_clip

            model_w = whisperx.load_model(wx_path, device, compute_type="float16")
            audio_source = whisperx.load_audio(video_path)
            result = model_w.transcribe(audio_source, batch_size=16)
            
            # Alignment
            align_model_dir = os.path.join(MODELS_DIR, "PyTorch")
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir=align_model_dir)
            result = whisperx.align(result["segments"], model_a, metadata, audio_source, device)
            
            all_words_global = []
            for s in result["segments"]:
                if 'words' in s: all_words_global.extend(s['words'])

            del model_w, model_a, audio_source, metadata
            clear_vram()
            
            # === SEGMENT SELECTION ===
            final_segments = []
            if enable_hard_cut:
                current_time_pointer = 0.0
                for s in result["segments"]:
                    if len(final_segments) >= max_clips: break
                    start_t = s['start']
                    if start_t >= current_time_pointer:
                        end_t = start_t + target_dur
                        if end_t <= video_duration:
                            custom_seg = s.copy(); custom_seg['end'] = end_t
                            final_segments.append(custom_seg)
                            current_time_pointer = end_t
            else:
                final_segments = [s for s in result["segments"] if (target_dur - tol_minus) <= (s['end'] - s['start']) <= (target_dur + tol_plus)]

            if not final_segments:
                st.warning("No segments found."); status_box.empty(); st.stop()

            # === PHASE 2: AUDIO CAPTIONING ===
            audio_captions_map = {}
            if enable_audio_cap:
                status_box.info(f"👂 **Phase 2/3: Audio Analysis ({len(final_segments)} clips)**...")
                a_engine = load_audio_engine()
                # Ensure loaded
                if not a_engine.model:
                    with st.spinner("Loading Audio Engine..."):
                        a_engine.load(log_callback=status_box.text)
                
                if a_engine.model:
                    prog_a = st.progress(0)
                    full_audio_clip = AudioFileClip(video_path)
                    for i, seg in enumerate(final_segments[:100]):
                        cut_end = seg['end'] if not enable_hard_cut else (seg['start'] + target_dur)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_aud:
                            sub_a = full_audio_clip.subclipped(seg['start'], cut_end)
                            sub_a.write_audiofile(tmp_aud.name, logger=None)
                            tmp_aud_path = tmp_aud.name
                        try:
                            audio_captions_map[i] = a_engine.caption_audio(tmp_aud_path, audio_prompt)
                        except Exception as e: 
                            print(f"Audio Cap Error: {e}")
                            audio_captions_map[i] = ""
                        finally: 
                            if os.path.exists(tmp_aud_path): os.unlink(tmp_aud_path)
                        prog_a.progress((i+1)/len(final_segments))
                    full_audio_clip.close(); 
                    # Optionally clear audio engine here to save VRAM for vision?
                    # a_engine.clear() 
                    clear_vram()
                else: st.error("Audio Engine Load Error.")

            # === PHASE 3: VISION & MERGE ===
            status_box.info("👁️ **Phase 3/3: Vision Captioning & Merging**...")
            folder_name = project_name.strip() if project_name.strip() else f"dataset_{target_dur}s"
            out_dir = os.path.join(BASE_DIR, folder_name); os.makedirs(out_dir, exist_ok=True)
            
            st.success(f"Found {len(final_segments)} clips. Saving to: {out_dir}") 
            
            v_engine = load_vision_engine()
            video_f = VideoFileClip(video_path)
            prog_v = st.progress(0)

            for i, seg in enumerate(final_segments[:100]):
                base = f"clip_{i+1:03d}"; c_path = os.path.join(out_dir, f"{base}.mp4")
                cut_end = seg['end'] if not enable_hard_cut else (seg['start'] + target_dur)
                
                sub = video_f.subclipped(seg['start'], cut_end)
                if not keep_orig: sub = sub.resized(new_size=(out_width, out_height)).write_videofile(c_path, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                else: sub.write_videofile(c_path, codec="libx264", audio_codec="aac", preset="medium", logger=None)
                
                # Streaming UI
                stream_box = st.empty()
                def on_token(text):
                    stream_box.markdown(f"**📝 Generating:** {text}")

                vis_cap = v_engine.caption(c_path, "video", lora_trigger, user_instruction, 
                                           gen_config=GEN_CONFIG, stream_callback=on_token)
                stream_box.empty()
                
                if enable_hard_cut:
                    valid_words = [w['word'] for w in all_words_global if w['start'] >= seg['start'] and w['end'] <= cut_end]
                    speech = " ".join(valid_words).strip()
                    if not speech: speech = seg['text'].strip()
                else: speech = seg['text'].strip()
                
                aud_cap = audio_captions_map.get(i, "").strip()
                if aud_cap: final_text = f"{vis_cap} In the background, {aud_cap}. The character says: \"{speech}\""
                else: final_text = f"{vis_cap} The character says: \"{speech}\""
                
                final_text = final_text.replace("..", ".").replace("  ", " ").strip()
                
                with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f: f.write(final_text)
                with st.expander(f"✅ {base}"): st.video(c_path); st.info(f"**Caption:** {final_text}")
                prog_v.progress((i+1)/len(final_segments))

            video_f.close(); 
            clear_vram()
            st.success("✅ DONE! v5.0 Pipeline Finished.")
            end_ts = time.time(); mins, secs = divmod(end_ts - start_ts, 60)
            timer_placeholder.success(f"⏱️ Total Time: {int(mins)}m {int(secs)}s")

        except Exception as e:
            st.error(f"Critical Error: {e}")
            import traceback; st.code(traceback.format_exc())

# =================================================================================================
# MODE 2: BULK VIDEO CAPTIONER
# =================================================================================================
elif app_mode == "📝 Bulk Video Captioner":
    if 'v_bulk_path' not in st.session_state: st.session_state['v_bulk_path'] = ""
    col_v, col_vbtn = st.columns([3, 1])
    with col_vbtn:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['v_bulk_path'] = sel; st.rerun()
    with col_v: v_dir = st.text_input("Folder Path:", value=st.session_state['v_bulk_path'])
    
    st.markdown("### 🛠️ Bulk Processing Options")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        enable_vis = st.checkbox("✅ Enable Vision Captioning (Qwen-VL/GGUF)", value=True)
    with col_opt2:
        enable_speech = st.checkbox("✅ Enable Speech Transcription (WhisperX)", value=True)

    if st.button("🚀 START BULK CAPTIONING") and os.path.exists(v_dir):
        if not enable_vis and not enable_speech:
            st.error("Select at least one option!")
        else:
            start_ts = time.time()
            status_box = st.empty()
            videos = [f for f in os.listdir(v_dir) if f.lower().endswith((".mp4", ".mkv"))]
            transcriptions = {}
            
            if not videos: st.warning("No videos found!")
            else:
                # 1. FAZA AUDIO (WHISPER)
                if enable_speech:
                    status_box.info(f"🎤 **Phase 1: Transcribing Audio for {len(videos)} clips...**")
                    try:
                        # Ensure model
                        wx_repo = "Systran/faster-whisper-large-v3"
                        wx_path = os.path.normpath(os.path.join(MODELS_DIR, wx_repo))
                        essential_wx = ["config.json", "model.bin", "tokenizer.json", "vocabulary.json"]
                        if not all(os.path.exists(os.path.join(wx_path, f)) for f in essential_wx):
                             with st.spinner("Ensuring Whisper Model..."):
                                 wx_path = download_model(wx_repo, MODELS_DIR, specific_files=essential_wx)

                        model_w = whisperx.load_model(wx_path, device, compute_type="float16")
                        prog_a = st.progress(0)
                        for i, v_name in enumerate(videos):
                            full_p = os.path.join(v_dir, v_name)
                            audio = whisperx.load_audio(full_p)
                            result = model_w.transcribe(audio, batch_size=16)
                            full_text = " ".join([seg['text'].strip() for seg in result['segments']])
                            transcriptions[v_name] = full_text
                            prog_a.progress((i+1)/len(videos))
                        del model_w, audio; clear_vram()
                    except Exception as e: st.error(f"Whisper Error: {e}"); clear_vram()
                
                # 2. FAZA VISION & MERGE
                if enable_vis:
                    status_box.info(f"👁️ **Phase 2: Visual Captioning...**")
                    v_engine = load_vision_engine()
                
                prog_main = st.progress(0)
                for i, v_name in enumerate(videos):
                    p = os.path.join(v_dir, v_name)
                    final_txt = ""
                    
                    # Vision Part
                    if enable_vis:
                        stream_box = st.empty()
                        def on_token(text):
                            stream_box.markdown(f"**Processing {v_name}:** {text}")

                        vis_cap = v_engine.caption(p, "video", lora_trigger, user_instruction, 
                                                   gen_config=GEN_CONFIG, stream_callback=on_token)
                        stream_box.empty()
                        final_txt += vis_cap
                    
                    # Speech Part
                    if enable_speech:
                        speech = transcriptions.get(v_name, "")
                        if speech:
                            separator = " The person says: " if enable_vis else "The person says: "
                            final_txt += f'{separator}"{speech}"'
                    
                    with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: 
                        f.write(final_txt.strip())
                    
                    prog_main.progress((i+1)/len(videos))
                
                if enable_vis:
                    # Should we clear? Not strictly necessary if we want to keep it warm, but safer for VRAM
                    # v_engine.clear() 
                    clear_vram()
                
                status_box.empty()
                st.success("✅ DONE! Bulk Processing finished.")
                mins, secs = divmod(time.time() - start_ts, 60)
                st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")

# =================================================================================================
# MODE 3: IMAGE FOLDER CAPTIONER
# =================================================================================================
else: 
    if 'img_path' not in st.session_state: st.session_state['img_path'] = ""
    col_p, col_b = st.columns([3, 1])
    with col_b:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['img_path'] = sel; st.rerun()
    with col_p: img_dir = st.text_input("Path:", value=st.session_state['img_path'])
    
    if st.button("🚀 CAPTION FOLDER") and os.path.exists(img_dir):
        start_ts = time.time()
        v_engine = load_vision_engine()
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".webp"))]
        
        prog = st.progress(0)
        for i, name in enumerate(imgs):
            p = os.path.join(img_dir, name)
            
            stream_box = st.empty()
            def on_token(text):
                stream_box.markdown(f"**{name}:** {text}")
                
            cap = v_engine.caption(p, "image", lora_trigger, user_instruction, 
                                   gen_config=GEN_CONFIG, stream_callback=on_token)
            stream_box.empty()
            
            with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: 
                f.write(cap)
            prog.progress((i+1)/len(imgs))
            
        st.success("✅ DONE! Folder finished.")
        mins, secs = divmod(time.time() - start_ts, 60)
        st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")

st.markdown("---")
st.markdown("<center><b>Project maintained by Cyberbol, Powered by FNGarvin</b></center>", unsafe_allow_html=True)