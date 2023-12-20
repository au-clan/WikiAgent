#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

# python src/run.py --multirun \
#     experiment=random \
#     model=gpt-3.5-turbo \
#     agent=link_click/gpt_dfs_back,link_click/gpt_dfs_back_reason \
#     agent.fewshot=2 \
#     run_name="${filename}" \
#     > "scripts/${filename}.log" 2>&1

python src/run.py --multirun \
    experiment=random \
    model=gpt-3.5-turbo \
    agent=link_click/gpt_dfs,link_click/gpt_dfs_reason \
    agent.fewshot=2 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

