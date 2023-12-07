
#!/bin/bash

#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

#MODELS=("llama_2_7b_chat_hf" "mistral_7b_instruct" "dolphin_2.2.1_mistral_7b" "mistral_7b_openorca" "openhermes_2.5_mistral_7b") 
model="llama_2_7b_chat_hf"
python src/run.py \
    --multirun model.device=cuda model.dtype=bfloat16 model.batch_size=2 \
    experiment=human_test_set \
    model="${model}" \
    agent=link_click/lmql_dfs_back_reminder,link_click/lmql_dfs_back_reminder_reason\
    agent.fewshot=0,1,2\
    run_name="human_test_set_${model}" \
    > "human_test_set_${model}.log" 2>&1
