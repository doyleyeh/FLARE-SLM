#!/usr/env/bin bash
# Install conda
# conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126    #torch 12.6, could modify to  11.8, 12.1, 12.4

# Install packages
pip install -r requirements.txt
python -m spacy download en_core_web_sm
conda install -c conda-forge cchardet # if you get an error about cchardet
######## For xLSTM packages
# # This transformers version is the one that has the XLSTM model if you want to use it (version is 4.47 so can't use gemma)
# pip install 'transformers @ git+https://github.com/NX-AI/transformers.git@integrate_xlstm'
# pip install xlstm mlstm_kernels


# elasticsearch disable disk usage threshold
#curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'
