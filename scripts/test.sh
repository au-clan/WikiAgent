
#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

python src/run.py experiment.num_agents=50  +agent.window_size=1000 model.batch_size=4 > navigate_jsonprompt.log 2>&1