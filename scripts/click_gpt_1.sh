#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py --multirun \
    experiment=click paths_real_file=/dlabdata1/charneca/WikiNav/data/wikispeedia/wikispeedia_paths-and-graph/top30_cur_tgt_1.tsv \
    experiment.agent_batch_size=1500 \
    model=gpt-3.5-turbo \
    agent=navigate/gpt_headings,navigate/gpt_headings_reason \
    agent.max_actions=1 \
    agent.fewshot=0 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

