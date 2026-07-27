#!/bin/bash
# AI Video Clipper & LoRA Captioner Launcher

# Defaults
PORT=8501
HOST="127.0.0.1"

# Parse args
while getopts ":p:h:" opt; do
  case $opt in
    p) PORT="$OPTARG"
    ;;
    h) HOST="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Set up local model paths
export HF_HOME="./models"
export TORCH_HOME="./models"
export NVIDIA_LIBS=$(python3 -c 'import site, os, glob; paths = [glob.glob(os.path.join(p, "nvidia/*/lib")) for p in site.getsitepackages()]; print(":".join([p for sub in paths for p in sub]))' 2>/dev/null)
export LD_LIBRARY_PATH=$NVIDIA_LIBS:$LD_LIBRARY_PATH
export KMP_DUPLICATE_LIB_OK=TRUE

# Activate environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Starting Streamlit on $HOST:$PORT..."
streamlit run app.py --server.port $PORT --server.address $HOST

