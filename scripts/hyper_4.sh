#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py \
    --multirun model.batch_size=6\
    experiment=random \
    model=llama_2_7b_chat_hf \
    agent=link_click/dfs_back \
    agent.fewshot=2 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

python src/run.py \
    --multirun model.batch_size=6\
    experiment=random \
    model=mistral_7b_instruct_v0.2 \
    agent=link_click/dfs_back \
    agent.fewshot=1 \
    run_name="${filename}" \
    > "scripts/${filename}.log"

