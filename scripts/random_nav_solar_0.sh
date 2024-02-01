#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py \
    --multirun model.batch_size=2\
    experiment=random \
    model=solar_10.7b_instruct_v1.0 \
    agent=navigate/lmql_headings,navigate/lmql_headings_reason \
    experiment.num_agents=100 experiment.agent_batch_size=100 \
    experiment.agent_batch_id=0\
    agent.fewshot=0 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

