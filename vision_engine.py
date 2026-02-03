#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner - GGUF Vision Engine
# Contributor: FNGarvin | License: MIT
# --------------------------------------------------------------------------------

import os
import gc
import threading
import torch
from downloader import download_model

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None
    Llava15ChatHandler = None

class GGUFVisionEngine:
    """
    Vision Engine powered by llama-cpp-python for GGUF models.
    Supports targeted weight/projector loading and streaming output.
    """
    def __init__(self, model_id, model_file=None, projector_file=None, device="cuda", models_dir="models"):
        self.model_id = model_id
        self.model_file = model_file
        self.projector_file = projector_file
        self.device = device
        self.models_dir = models_dir
        
        self.llm = None
        self.chat_handler = None

    def load(self, log_callback=None):
        """Loads the GGUF model and its vision projector."""
        if Llama is None:
            raise ImportError("llama-cpp-python not found. Please install it with CUDA support.")

        # 1. Resolve Path & Download if needed
        dest_path = os.path.join(self.models_dir, self.model_id)
        
        # If files aren't specified, we can't really guess them reliably from here 
        # unless we scan the folder after a full sync. 
        # For now, we assume app.py provides them or we trigger a sync.
        needed = []
        if self.model_file: needed.append(self.model_file)
        if self.projector_file: needed.append(self.projector_file)
        
        if not os.path.exists(dest_path) or (needed and not all(os.path.exists(os.path.join(dest_path, f)) for f in needed)):
            download_model(self.model_id, self.models_dir, specific_files=needed, log_callback=log_callback)

        model_path = os.path.join(dest_path, self.model_file)
        projector_path = os.path.join(dest_path, self.projector_file) if self.projector_file else None

        if log_callback: log_callback(f"🧠 Loading GGUF: {self.model_file}...")
        
        # 2. Initialize Backend
        self.chat_handler = Llava15ChatHandler(clip_model_path=projector_path) if projector_path else None
        
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=2048, # Sufficient for vision + prompt
            n_gpu_layers=-1, # All on GPU
            verbose=False
        )
        
        if log_callback: log_callback("✅ Vision Engine Ready (GGUF).")

    def caption(self, content_path, content_type, trigger, instruction, gen_config=None, stream_callback=None):
        """
        Generates a caption for an image or video path.
        Supports salvaged streaming iterator logic.
        """
        if not self.llm:
            raise RuntimeError("Engine not loaded. Call load() first.")

        # 1. Format prompt
        if not instruction.strip():
            instruction = "Describe this {type} in high detail."
        
        final_prompt = instruction.replace("{trigger}", trigger if trigger else "")
        final_prompt = final_prompt.replace("{type}", "video" if content_type == "video" else "image")
        
        # 2. Build message for Llava-style vision
        # Note: llama-cpp-python typically expects base64 or file paths in the content block
        # depending on the chat handler. Llava15ChatHandler handles the heavy lifting.
        import base64
        def get_base64_image(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # For video, we sample the middle frame for now as llama.cpp vision is often frame-based
        if content_type == "video":
            from moviepy.video.io.VideoFileClip import VideoFileClip
            video = VideoFileClip(content_path)
            temp_img = content_path + ".thumb.jpg"
            video.save_frame(temp_img, t=video.duration/2)
            img_b64 = get_base64_image(temp_img)
            video.close()
            try: os.remove(temp_img)
            except: pass
        else:
            img_b64 = get_base64_image(content_path)

        messages = [
            {"role": "system", "content": "You are a professional video captioner and AI data labeller."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": final_prompt}
            ]}
        ]

        # 3. Generation Logic (Streaming vs Blocking)
        gen_params = {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if gen_config:
            gen_params.update(gen_config)

        if stream_callback:
            # Salvaged Streaming Logic
            response_iter = self.llm.create_chat_completion(
                messages=messages,
                stream=True,
                **gen_params
            )
            
            full_text = ""
            for chunk in response_iter:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        full_text += content
                        stream_callback(self._post_process(full_text, trigger))
            
            return self._post_process(full_text, trigger)
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                **gen_params
            )
            output_text = response["choices"][0]["message"]["content"]
            return self._post_process(output_text, trigger)

    def _post_process(self, text, trigger):
        """Humble post-processing to ensure trigger word is included."""
        text = text.replace("**", "").replace("##", "").strip()
        if trigger and trigger.lower() not in text.lower():
            text = f"{trigger}, {text}"
        return text

    def clear(self):
        """Aggressive resource cleanup."""
        self.llm = None
        self.chat_handler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# EOF vision_engine.py
