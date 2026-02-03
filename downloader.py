#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner - High Performance Downloader
# Contributor: FNGarvin | License: MIT
# --------------------------------------------------------------------------------

import os
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
CHUNK_SIZE = 1024 * 1024  # 1MB buffer for streaming
MULTI_CONN_THRESHOLD = 50 * 1024 * 1024  # 50MB: Files larger than this use multi-connection
CONNS_PER_FILE = 8  # Simultaneous connections for a single large file
MAX_PARALLEL_FILES = 4  # How many files to start at once

def list_repo_files(model_id):
    """
    Fetch file list from HF API directly (bypassing hf_hub dependency/offline checks).
    """
    url = f"https://huggingface.co/api/models/{model_id}/tree/main?recursive=true"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [item["path"] for item in data if item.get("type") == "file"]
    except Exception as e:
        raise RuntimeError(f"Failed to list repo files for {model_id}: {e}")

def download_segment(url, local_path, start, end):
    """Download a specific byte range of a file."""
    headers = {"Range": f"bytes={start}-{end}"}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(local_path, "r+b") as f:
            f.seek(start)
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False

def download_single_file(url, local_path, log_callback=None):
    """Download a file with multi-connection support and atomic .part renaming."""
    
    def safe_log(msg):
        if log_callback:
            try: log_callback(msg)
            except: pass

    part_path = local_path + ".part"
    filename = os.path.basename(local_path)
    
    try:
        # 1. Get file size and check range support
        res = requests.head(url, allow_redirects=True, timeout=30)
        total_size = int(res.headers.get("content-length", 0))
        accept_ranges = res.headers.get("accept-ranges") == "bytes"
        
        if total_size == 0:
            res = requests.get(url, stream=True, timeout=30)
            total_size = int(res.headers.get("content-length", 0))
            
        # 2. Decide: Single vs Multi-connection
        if accept_ranges and total_size > MULTI_CONN_THRESHOLD:
            safe_log(f"   🚀 Large file: {filename} ({total_size/(1024*1024):.1f}MB). Using {CONNS_PER_FILE} conns.")
            
            with open(part_path, "wb") as f:
                f.truncate(total_size)
            
            segment_size = total_size // CONNS_PER_FILE
            futures = []
            with ThreadPoolExecutor(max_workers=CONNS_PER_FILE) as executor:
                for i in range(CONNS_PER_FILE):
                    start = i * segment_size
                    end = start + segment_size - 1 if i < CONNS_PER_FILE - 1 else total_size - 1
                    futures.append(executor.submit(download_segment, url, part_path, start, end))
                
                for future in as_completed(futures):
                    if not future.result():
                        raise RuntimeError(f"Segment download failed for {filename}")
        else:
            if total_size > 1024*1024:
                safe_log(f"   📥 Downloading {filename}...")
                
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        
        # 3. Atomic Rename
        if os.path.exists(local_path):
            os.remove(local_path)
        os.rename(part_path, local_path)
        return True
        
    except Exception as e:
        safe_log(f"   ❌ Failed: {filename}: {e}")
        if os.path.exists(part_path):
            try: os.remove(part_path)
            except: pass
        return False

def download_model(model_id, target_base_dir, specific_files=None, log_callback=None):
    """
    Surgical Downloader: Parallel files + Multi-connection single-file sharding.
    Supports targeted file selection to avoid downloading entire repos.
    """
    def log(msg):
        print(msg)
        if log_callback:
            try: log_callback(msg)
            except: pass

    dest_path = os.path.join(target_base_dir, model_id)
    os.makedirs(dest_path, exist_ok=True)
    
    # Offline-First: Check complete flag
    complete_flag = os.path.join(dest_path, ".download_complete")
    if os.path.exists(complete_flag):
        # Even if flag exists, if specific_files are requested, verify they exist
        if not specific_files or all(os.path.exists(os.path.join(dest_path, f)) for f in specific_files):
            log(f"✅ Model {model_id} up-to-date.")
            return dest_path

    log(f"⚙️ Syncing: {model_id}")

    try:
        # 1. Get file list via direct API (Bypasses hf_hub offline triggers)
        available_files = list_repo_files(model_id)
        
        if specific_files:
            files_to_download = [f for f in available_files if f in specific_files]
        else:
            exclude = [".git", ".gitattributes", ".download_complete"]
            files_to_download = [f for f in available_files if not any(x in f for x in exclude)]
        
        # 2. Processing
        total_files = len(files_to_download)
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES) as executor:
            future_to_file = {}
            for filename in files_to_download:
                file_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
                local_file_path = os.path.join(dest_path, filename)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                if os.path.exists(local_file_path):
                    success_count += 1
                    continue
                
                future_to_file[executor.submit(download_single_file, file_url, local_file_path, log_callback)] = filename

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                if future.result():
                    success_count += 1
                    log(f"   ✅ Verified: {filename} ({success_count}/{total_files})")
                else:
                    log(f"   ⚠️ Warning: {filename} failed.")

        if success_count == total_files:
            with open(complete_flag, "w") as f:
                f.write(f"verified_{time.time()}")
            log(f"✨ Model {model_id} verified 100%.")
            return dest_path
        else:
            raise RuntimeError(f"Incomplete: {success_count}/{total_files}")
        
    except Exception as e:
        log(f"❌ Transfer Failed: {e}")
        # If offline, we return the path anyway and hope for the best
        return dest_path

# EOF downloader.py
