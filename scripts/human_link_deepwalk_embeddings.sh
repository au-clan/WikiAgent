#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py \
    --multirun \
    experiment=human_test_set \
    agent=link_click/embedding_deepwalk_dfs \
    experiment.num_agents=1914 experiment.agent_batch_size=1914\
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

