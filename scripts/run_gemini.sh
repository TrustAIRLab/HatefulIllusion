DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

eval "$(conda shell.bash hook)"
conda create -n gemini python=3.10 -y
conda activate gemini
pip install numpy==1.26.4
pip install pandas
pip install -q -U google-generativeai
pip install tiktoken
pip install transformers_stream_generator
pip install qwen-vl-utils
pip install transformers -U
pip install diffusers -U
pip install open-clip-torch
pip install opencv-python==4.5.5.64
pip install datasets
conda activate gemini

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

API_KEY=

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name gemini_2 \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target \
    --api_key $API_KEY
done