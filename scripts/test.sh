
#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status
export HOME=/dlabdata1/charneca
source activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

#MODELS=("llama_2_7b_chat_hf" "mistral_7b_instruct" "dolphin_2.2.1_mistral_7b" "mistral_7b_openorca" "openhermes_2.5_mistral_7b") 
MODELS=("llama_2_13b_chat_hf")
for model in "${MODELS[@]}"; do
    python src/run.py \
        experiment=random \
        experiment.override=True \
        experiment.num_agents=5 \
        agent.fewshot=3 \
        agent.max_actions=3 \
        model.device=cuda \
        model.dtype=bfloat16 \
        model.batch_size=1 \
        +model.lmql.use_flash_attention_2=True \
        model="${model}" \
        agent=link_click/lmql_dfs_back_reminder_reason\
        > output.log 2>&1
done



# python src/run.py \
#     --multirun model.device=cuda model.batch_size=1 \
#     model=mistral_7b_openorca model.dtype=bfloat16 \
#     agent=link_click/lmql_dfs_back_reminder_reason \
#     agent.fewshot=0,1,2,3,4 \
#     run_name=link_click_reason \
#     > link_click_reason.log 2>&1


#same script but a for loop that runs the same src/run.py for each model
# command to test batch size
# python src/run.py --multirun model.device=cuda model.batch_size=2 model.dtype=bfloat16 agent=link_click/lmql_dfs_back_reminder_reason experiment.override=True experiment.num_agents=5 agent.max_actions=1 agent.fewshot=0,1,2,3,4
