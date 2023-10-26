
#!/bin/bash


set -e # Exit immediately if a command exits with a non-zero status

conda activate miniconda3/envs/wikinav || { echo "Failed to activate conda environment" > test.log; exit 1; }

cd WikiNav
echo "Running Wikinav" > test.log
python WikiNav/src/run.py \
    --multirun model.device=cuda model.half_precision=True model.batch_size=1 \
    agent=link_click_dfs_back_completion_agent run_name=completion_test experiment.num_agents=1 \
    > completion_test.log 2>&1

