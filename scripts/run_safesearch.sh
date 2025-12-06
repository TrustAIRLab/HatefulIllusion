DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export LD_LIBRARY_PATH=""
source /opt/conda/bin/activate

pip install google-cloud
pip install google-cloud-vision
source /opt/conda/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# make sure you have downloaded the credential file at .env/google_cloud.json

for target in hate_slangs hate_symbols
do
    python $DIR/detect.py \
    --model_name SafeSearch \
    --dataset_name illusion \
    --query_mode zero-shot \
    --target $target
done
