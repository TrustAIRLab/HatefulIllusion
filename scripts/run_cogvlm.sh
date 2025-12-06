DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export LD_LIBRARY_PATH=""
source /opt/conda/bin/activate
pip install torch==2.1.0 transformers==4.35.0 accelerate==0.24.1 sentencepiece==0.1.99 einops==0.7.0 xformers==0.0.22.post7 triton==2.1.0 pandas openpyxl
pip install diffusers -U
pip install open-clip-torch
pip install opencv-python==4.5.5.64

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name cogvlm \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target
done