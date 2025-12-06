DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

cd $DIR
source /opt/conda/bin/activate
pip install openai
pip install diffusers -U
pip install open-clip-torch
pip install opencv-python==4.5.5.64

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

API_KEY=

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name omni \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target \
    --api_key $API_KEY
done