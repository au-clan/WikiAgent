#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py \
    --multirun model.batch_size=3\
    experiment=random \
    model=mistral_7b_instruct_v0.2 \
    agent=link_click/lmql_dfs_back,link_click/lmql_dfs_back_reason \
    agent.fewshot=1 \
    model.temperature=0.2,0.5,1,1.5 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

