#!/usr/env/bin bash

conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# elasticsearch disable disk usage threshold
#curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'
