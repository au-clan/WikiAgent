#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py \
    --multirun model.batch_size=3\
    experiment=human_test_set \
    model=mistral_7b_instruct_v0.2 \
    agent=link_click/lmql_dfs_back,link_click/lmql_dfs_back_reason \
    experiment.agent_batch_id=1\
    agent.fewshot=1 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

