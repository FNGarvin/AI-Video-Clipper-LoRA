# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner (v3.7)
# 🏆 CREDITS: Cyberbol (Logic), FNGarvin (Engine), WildSpeaker (5090 Fix)
# --------------------------------------------------------------------------------

import os
import sys
import streamlit as st
import whisperx
from moviepy import VideoFileClip
import tempfile
import torch
import gc
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import tkinter as tk
from tkinter import filedialog

# --- 1. BOOTSTRAP & PATCHES ---
try:
    import patches
    patches.apply_patches()
except ImportError:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import torch.serialization
    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata, Node
        import typing
        torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, Node, typing.Any])
    except: pass
    if not hasattr(torch, "_patched_for_5090"):
        _orig_load = torch.load
        def _safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        torch.load = _safe_load
        torch._patched_for_5090 = True

# --- 2. KONFIGURACJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR

# --- 3. UI CONFIG ---
st.set_page_config(page_title="AI Clipper v3.7", layout="wide")
st.title("👁️🐧 AI Video Clipper & LoRA Captioner")
st.markdown("v3.7 | Created by: **Cyberbol** | Engine: **FNGarvin** (UV) | 5090 Fix: **WildSpeaker**")

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

def scan_local_gguf_models(models_dir):
    """
    Scans the models directory for GGUF model pairs:
    A pair consists of a .gguf file (weights) and a mmproj-*.gguf file (projector).
    Returns a dictionary of found models formatted for model_options.
    """
    discovered = {}
    if not os.path.exists(models_dir):
        return discovered
        
    for root, dirs, files in os.walk(models_dir):
        # We look for files ending in .gguf
        # Main model: NOT starting with mmproj-
        # Projector: STARTING with mmproj-
        ggufs = [f for f in files if f.endswith(".gguf") and not f.lower().startswith("mmproj-")]
        projectors = [f for f in files if f.lower().startswith("mmproj-") and f.endswith(".gguf")]
        
        if ggufs and projectors:
            # We use the relative path from MODELS_DIR as the repo/subfolder name
            rel_root = os.path.relpath(root, models_dir).replace("\\", "/")
            for g in ggufs:
                # If there's multiple projectors, we pick the first one 
                # (usually there's only one relevant one in a specific subfolder)
                label = f"🔍 Discovered: {os.path.basename(root)} ({g})"
                discovered[label] = {
                    "backend": "gguf",
                    "repo": rel_root, 
                    "model": g,
                    "projector": projectors[0]
                }
    return discovered

st.sidebar.divider()
# --- WYBÓR MODELU ---
model_options = {
    "GGUF: Gemma-3-12B (Next-Gen, 4-bit GGUF)": {
        "backend": "gguf",
        "repo": "unsloth/gemma-3-12b-it-GGUF",
        "model": "gemma-3-12b-it-IQ4_XS.gguf",
        "projector": "mmproj-F16.gguf"
    },
    "GGUF: Qwen3-VL-7B (Next-Gen, 4-bit GGUF)": {
        "backend": "gguf",
        "repo": "unsloth/Qwen3-VL-7B-Instruct-GGUF",
        "model": "Qwen3-VL-7B-Instruct-Q4_K_M.gguf",
        "projector": "mmproj-model-f16.gguf"
    },
    "Legacy: Qwen2-VL-7B (Legacy Transformers)": {
        "backend": "transformers",
        "id": "Qwen/Qwen2-VL-7B-Instruct"
    },
    "Legacy: Qwen2-VL-2B (Legacy Transformers)": {
        "backend": "transformers",
        "id": "Qwen/Qwen2-VL-2B-Instruct"
    }
}

# Auto-Discovery of Local GGUF Models
local_ggufs = scan_local_gguf_models(MODELS_DIR)
if local_ggufs:
    # Filter out models already in hardcoded options to avoid duplicates
    existing_models = [m["model"] for m in model_options.values() if m.get("backend") == "gguf"]
    for label, config in local_ggufs.items():
        if config["model"] not in existing_models:
            model_options[label] = config

model_label = st.sidebar.radio("Vision Model:", list(model_options.keys()), index=0)
SELECTED_MODEL = model_options[model_label]

# --- SIDEBAR: ADVANCED OPTIONS ---
st.sidebar.divider()
with st.sidebar.expander("🛠️ Advanced Generation Options"):
    gen_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    gen_top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    gen_max_tokens = st.number_input("Max New Tokens", 64, 2048, 256)
    
GEN_CONFIG = {
    "temperature": gen_temp,
    "top_p": gen_top_p,
    "max_tokens": gen_max_tokens # unified key for both engines to map from
}

st.sidebar.divider()
st.sidebar.markdown("### 📝 Vision Instructions")
default_prompt = "Describe this {type} in detail for a dataset. Main subject: {trigger}. Describe the action, camera movement, lighting, atmosphere, and background."
user_instruction = st.sidebar.text_area("System Prompt:", value=default_prompt, height=150)

# --- 4. FUNKCJE POMOCNICZE ---
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def load_vision_models():
    if SELECTED_MODEL["backend"] == "gguf":
        # Handle GGUF Loading via st.session_state for persistence
        if 'vision_engine' not in st.session_state or st.session_state.get('last_model') != str(SELECTED_MODEL):
            if 'vision_engine' in st.session_state:
                st.session_state['vision_engine'].clear()
            
            from vision_engine import GGUFVisionEngine
            with st.status(f"🚀 Preparing GGUF Engine...", expanded=True) as status:
                engine = GGUFVisionEngine(
                    SELECTED_MODEL["repo"], 
                    model_file=SELECTED_MODEL["model"],
                    projector_file=SELECTED_MODEL["projector"],
                    device=device,
                    models_dir=MODELS_DIR
                )
                engine.load(log_callback=status.write)
                status.update(label="✅ GGUF Engine Ready!", state="complete", expanded=False)
            st.session_state['vision_engine'] = engine
            st.session_state['last_model'] = str(SELECTED_MODEL)
        return st.session_state['vision_engine'], None
    else:
        # Legacy Transformers Path
        model_id = SELECTED_MODEL["id"]
        st.info(f"⏳ Loading Vision Engine ({model_id})...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa", low_cpu_mem_usage=True 
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

def select_folder_dialog():
    root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root); root.destroy()
    return folder_path

def caption_content(content_path, content_type, model, processor, trigger, custom_prompt):
    if not custom_prompt.strip(): custom_prompt = "Describe this content in high detail."
    
    # GGUF Backend Routing
    if SELECTED_MODEL["backend"] == "gguf":
        # Pass parameters to GGUF engine
        gguf_engine = model # In GGUF mode, 'model' is the GGUFVisionEngine instance
        # Map parameters (GGUF uses 'max_tokens' instead of 'max_new_tokens')
        return gguf_engine.caption(content_path, content_type, trigger, custom_prompt, gen_config=GEN_CONFIG)

    # Legacy Transformers Backend
    model_id = SELECTED_MODEL["id"]
    if "2B" in model_id: custom_prompt += " Output as a single, continuous paragraph. No markdown."
    final_prompt = custom_prompt.replace("{trigger}", trigger if trigger else "").replace("{type}", "video" if content_type == "video" else "image")
    
    messages = [{"role": "user", "content": [{"type": content_type, content_type: content_path, "max_pixels": 360*420, "fps": 1.0}, {"type": "text", "text": final_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=GEN_CONFIG["max_tokens"],
        temperature=GEN_CONFIG["temperature"],
        top_p=GEN_CONFIG["top_p"],
        do_sample=GEN_CONFIG["temperature"] > 0
    )
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    output_text = output_text.replace("**", "").replace("##", "")
    if trigger and trigger not in output_text: output_text = f"{trigger}, {output_text}"
    return output_text

# --- 5. LOGIKA UI ---
lora_trigger = st.text_input("LoRA Trigger Word (Optional)", value="cbrl man")

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
    
    t_col1, t_col2 = st.columns(2)
    with t_col1: tol_minus = st.number_input("Tolerance Margin - (sec)", 0.0, 5.0, 0.0)
    with t_col2: tol_plus = st.number_input("Tolerance Margin + (sec)", 0.0, 10.0, 0.5)
    
    # --- NOWA SEKCJA Z LICZNIKIEM ---
    # Używamy kolumn, aby umieścić licznik obok przycisku
    col_btn, col_timer = st.columns([1, 4])
    with col_btn:
        start_processing = st.button("🚀 START PROCESSING")
    with col_timer:
        timer_placeholder = st.empty() # To jest miejsce na Twój licznik
    # --------------------------------

    if uploaded_file and start_processing:
        start_ts = time.time() # START CZASU
        timer_placeholder.info("⏱️ Processing started...")
        
        status_box = st.empty()
        status_box.info("🚀 **Phase 1/2: Initializing Audio Analysis...** WhisperX is starting up.")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read()); video_path = tmp.name
        try:
            status_box.info("⏳ **Phase 1/2: Analyzing Speech & Timestamps...** Finding the best clips.")
            
            # --- FAZA WHISPER (Performance Fix) ---
            model_w = whisperx.load_model("large-v3", device, compute_type="float16", download_root=MODELS_DIR)
            audio = whisperx.load_audio(video_path)
            result = model_w.transcribe(audio, batch_size=16)
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device)
            
            # CLEANUP
            del model_w, model_a, audio, metadata
            clear_vram()
            
            segments = [s for s in result["segments"] if (target_dur - tol_minus) <= (s['end'] - s['start']) <= (target_dur + tol_plus)]
            
            if segments:
                status_box.empty()
                folder_name = project_name.strip() if project_name.strip() else f"dataset_{target_dur}s"
                out_dir = os.path.join(BASE_DIR, folder_name); os.makedirs(out_dir, exist_ok=True)
                
                v_model, v_proc = load_vision_models()
                video_f = VideoFileClip(video_path)
                st.success(f"Found {len(segments)} clips. Saving to: {out_dir}")
                prog = st.progress(0)
                
                for i, seg in enumerate(segments[:100]):
                    base = f"clip_{i+1:03d}"; c_path = os.path.join(out_dir, f"{base}.mp4")
                    sub = video_f.subclipped(seg['start'], seg['end'])
                    
                    if not keep_orig:
                        sub = sub.resized(new_size=(out_width, out_height))
                        sub.write_videofile(c_path, codec="libx264", audio_codec="aac", fps=out_fps, preset="medium", logger=None)
                    else:
                        sub.write_videofile(c_path, codec="libx264", audio_codec="aac", preset="medium", logger=None)
                    
                    cap = caption_content(c_path, "video", v_model, v_proc, lora_trigger, user_instruction)
                    speech = seg['text'].strip()
                    with open(os.path.join(out_dir, f"{base}.txt"), "w", encoding="utf-8") as f: f.write(f"{cap} The person says: \"{speech}\"")
                    with st.expander(f"✅ {base}"):
                        st.video(c_path); st.info(f"💬 **Speech:** {speech}")
                    prog.progress((i+1)/len(segments))
                
                video_f.close()
                del v_model, v_proc
                clear_vram()
                
                st.success("✅ DONE! Processing finished successfully.")
                
                # --- KONIEC CZASU I WYNIK ---
                end_ts = time.time()
                total_seconds = end_ts - start_ts
                mins, secs = divmod(total_seconds, 60)
                timer_placeholder.success(f"⏱️ Total Time: {int(mins)}m {int(secs)}s")
                # -----------------------------
            else:
                st.warning("No segments found matching those exact duration margins.")
                status_box.empty()
                timer_placeholder.empty()
        finally:
            clear_vram(); 
            if os.path.exists(video_path): os.unlink(video_path)

elif app_mode == "📝 Bulk Video Captioner":
    if 'v_bulk_path' not in st.session_state: st.session_state['v_bulk_path'] = ""
    col_v, col_vbtn = st.columns([3, 1])
    with col_vbtn:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['v_bulk_path'] = sel; st.rerun()
    with col_v: v_dir = st.text_input("Folder Path:", value=st.session_state['v_bulk_path'])
    if st.button("🚀 START BULK CAPTIONING") and os.path.exists(v_dir):
        start_ts = time.time() # Start czasu dla Bulk
        v_model, v_proc = load_vision_models()
        videos = [f for f in os.listdir(v_dir) if f.lower().endswith((".mp4", ".mkv"))]
        prog = st.progress(0)
        for i, v_name in enumerate(videos):
            p = os.path.join(v_dir, v_name); cap = caption_content(p, "video", v_model, v_proc, lora_trigger, user_instruction)
            with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: f.write(cap)
            prog.progress((i+1)/len(videos))
        
        del v_model, v_proc
        st.success("✅ DONE! Bulk Captioning finished.")
        
        # Czas dla Bulk
        total_seconds = time.time() - start_ts
        mins, secs = divmod(total_seconds, 60)
        st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")
        clear_vram()

else: # IMAGE CAPTIONER
    if 'img_path' not in st.session_state: st.session_state['img_path'] = ""
    col_p, col_b = st.columns([3, 1])
    with col_b:
        if st.button("📂 Select Folder"):
            sel = select_folder_dialog()
            if sel: st.session_state['img_path'] = sel; st.rerun()
    with col_p: img_dir = st.text_input("Path:", value=st.session_state['img_path'])
    if st.button("🚀 CAPTION FOLDER") and os.path.exists(img_dir):
        start_ts = time.time() # Start czasu dla Images
        v_model, v_proc = load_vision_models()
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".webp"))]
        prog = st.progress(0)
        for i, name in enumerate(imgs):
            p = os.path.join(img_dir, name); cap = caption_content(p, "image", v_model, v_proc, lora_trigger, user_instruction)
            with open(os.path.splitext(p)[0] + ".txt", "w", encoding="utf-8") as f: f.write(cap)
            prog.progress((i+1)/len(imgs))
        
        del v_model, v_proc
        st.success("✅ DONE! Folder Captioning finished.")
        
        # Czas dla Images
        total_seconds = time.time() - start_ts
        mins, secs = divmod(total_seconds, 60)
        st.info(f"⏱️ Total Execution Time: {int(mins)}m {int(secs)}s")
        clear_vram()

st.markdown("---")
st.markdown("<center><b>Project maintained by Cyberbol</b></center>", unsafe_allow_html=True)