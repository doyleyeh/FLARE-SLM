#!/usr/bin/env bash
set -e

debug=true

source keys.sh
num_keys=${#keys[@]}

dataset=$1
config_file=$2

config_filename=$(basename -- "${config_file}")
config_filename="${config_filename%.*}"

debug_batch_size=1
batch_size=1
model=llama3.1-8b-i
# model=mamba2-i
# model=phi3.5-4b-i
# model=qwen2.5-7b
# model=gemma3-12b-i
# model=xlstm7b

temperature=0

# output=output/${dataset}/${model}/${config_filename}_single_retrieval.jsonl
# output=output/${dataset}/${model}/${config_filename}_no_retrieval.jsonl
# output=output/${dataset}/${model}/${config_filename}_top4.jsonl
# output=output/${dataset}/${model}/${config_filename}_top6.jsonl
# output=output/${dataset}/${model}/${config_filename}_top8.jsonl
# output=output/${dataset}/${model}/${config_filename}_0.8.jsonl
# output=output/${dataset}/${model}/${config_filename}_0.6.jsonl
output=output/${dataset}/${model}/${config_filename}.jsonl
echo 'output to:' $output

prompt_type=""
if [[ ${dataset} == '2wikihop' ]]; then
    input="--input data/2wikimultihopqa"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=512
elif [[ ${dataset} == 'strategyqa' ]]; then
    input="--input data/strategyqa/dev_beir"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=6
    max_num_examples=229
    max_generation_len=256
elif [[ ${dataset} == 'asqa' ]]; then
    prompt_type="--prompt_type general_hint_in_output"
    input="--input data/asqa/ASQA.json"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'asqa_hint' ]]; then
    prompt_type="--prompt_type general_hint_in_input"
    dataset=asqa
    input="--input data/asqa/ASQA.json"
    engine=elasticsearch
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'wikiasp' ]]; then
    input="--input data/wikiasp"
    engine=bing
    index_name=wikiasp
    fewshot=4
    max_num_examples=500
    max_generation_len=512
else
    exit
fi

# query api
if [[ ${debug} == "true" ]]; then
    python -m src.openai_api \
        --model ${model} \
        --dataset ${dataset} ${input} ${prompt_type} \
        --config_file ${config_file} \
        --fewshot ${fewshot} \
        --search_engine ${engine} \
        --index_name ${index_name} \
        --max_num_examples 10 \
        --max_generation_len ${max_generation_len} \
        --batch_size ${debug_batch_size} \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0 \
        --openai_keys ${test_key} \
        --debug
    exit
fi

function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

joined_keys=$(join_by " " "${keys[@]:0:${num_keys}}")
python -m src.openai_api \
    --model ${model} \
    --dataset ${dataset} ${input} ${prompt_type} \
    --config_file ${config_file} \
    --fewshot ${fewshot} \
    --search_engine ${engine} \
    --index_name ${index_name} \
    --max_num_examples ${max_num_examples} \
    --max_generation_len ${max_generation_len} \
    --temperature ${temperature} \
    --batch_size ${batch_size} \
    --output ${output} \
    --num_shards 1 \
    --shard_id 0 \
    --openai_keys ${joined_keys} \
