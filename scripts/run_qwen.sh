DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export LD_LIBRARY_PATH=""

source /opt/conda/bin/activate
pip install transformers -U
pip install accelerate -U
pip install qwen_vl_utils
pip install flash-attn --no-build-isolation
pip install jinja2==3.1.0
pip install diffusers -U
pip install open-clip-torch
pip install opencv-python==4.5.5.64

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name qwen \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target
done
