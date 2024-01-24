#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py --multirun \
    experiment=random \
    model=gpt-4-turbo \
    agent=navigate/gpt_headings,navigate/gpt_headings_reason \
    experiment.num_agents=100 experiment.agent_batch_size=100 \
    experiment.agent_batch_id=0 \
    agent.fewshot=0 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1
