#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# get filename without the .sh extension
filename=$(basename -- "$0"); extension="${filename##*.}"; filename="${filename%.*}"

python src/run.py --multirun \
    experiment=click paths_real_file=/dlabdata1/charneca/WikiAgent/data/wikispeedia/wikispeedia_paths-and-graph/top30_cur_tgt.tsv \
    experiment.agent_batch_size=30 experiment.num_agents=30 \
    agent=link_click/embedding_openai_dfs \
    agent.max_actions=1 \
    agent.fewshot=0 \
    run_name="${filename}" \
    > "scripts/${filename}.log" 2>&1

