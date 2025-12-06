DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export LD_LIBRARY_PATH=""
pip install azure-ai-contentsafety
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

API_KEY=
ENDPOINT=

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name azure_multimodal \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target \
    --api_key $API_KEY \
    --endpoint $ENDPOINT
done