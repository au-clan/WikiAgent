#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py \
    --multirun model.batch_size=3\
    experiment=random \
    model=llama_2_7b_chat_hf \
    agent=link_click/lmql_dfs_back_reminder_reason \
    agent.fewshot=0,1,2,3 \
    run_name="${filename}" \
    > "${filename}.log" 2>&1

# python src/run.py \
#     --multirun model.batch_size=3\
#     experiment=random \
#     model=llama_2_7b_chat_hf \
#     agent=link_click/lmql_dfs_back_reminder,link_click/lmql_dfs_back_reminder_reason \
#     agent.fewshot=0,1,2,3 \
#     run_name="${filename}" \
#     > "${filename}.log" 2>&1
