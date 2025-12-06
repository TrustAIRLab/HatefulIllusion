DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

cd $DIR
source /opt/conda/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name safety_checker \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target
done