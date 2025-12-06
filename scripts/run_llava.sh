DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

cd $DIR
source /opt/conda/bin/activate
pip install -q diffusers transformers accelerate xformers qrcode
pip install -q diffusers transformers accelerate xformers qrcode
pip install opencv-python==4.5.5.64
pip install transformers -U
pip install accelerate==0.26.0
pip install jinja2==3.1.0
pip install diffusers -U
pip install open-clip-torch
source /opt/conda/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name llava \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target
done
