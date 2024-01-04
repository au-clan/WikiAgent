#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

# python src/run.py \
#     --multirun model.batch_size=2\
#     experiment=human_test_set \
#     model=mistral_7b_instruct_v0.2 \
#     agent=navigate/lmql_headings,navigate/lmql_headings_reason \
#     experiment.agent_batch_id=4\
#     agent.fewshot=0 \
#     run_name="${filename}" \
#     > "scripts/${filename}.log" 2>&1

python src/run.py \
    --multirun model.batch_size=2\
    experiment=human_test_set \
    model=mistral_7b_instruct_v0.2 \
    agent=navigate/lmql_headings \
    experiment.agent_batch_id=4\
    restart_from_agents_file=/dlabdata1/charneca/WikiNav/logs/inference/multiruns/human_nav_4/2023-12-27_13-36-54/0/agent_runs.tsv \
    agent.fewshot=0 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

python src/run.py \
    --multirun model.batch_size=2\
    experiment=human_test_set \
    model=mistral_7b_instruct_v0.2 \
    agent=navigate/lmql_headings_reason \
    experiment.agent_batch_id=4\
    agent.fewshot=0 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1
