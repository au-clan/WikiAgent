
#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status

source ~/.conda/etc/profile.d/conda.sh #|| { echo "Failed to source conda" > test.log; exit 1; }
conda activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

MODELS=("mistral_7b_openorca" "openhermes_2.5_mistral_7b" "dolphin_2.2.1_mistral_7b")
for model in "${MODELS[@]}"; do
    python src/run.py \
        --multirun model.device=cuda model.half_precision=True model.batch_size=2 \
        model="${model}" \
        agent=link_click/lmql_dfs_back_reminder \
        agent.fewshot=0,1,2,3,4,5\
        run_name="${model}_fewshot" \
        > "${model}_fewshot.log" 2>&1
done

# python src/run.py \
#     --multirun model.device=cuda model.half_precision=True model.batch_size=2 \
#     model=mistral_7b_instruct,mistral_7b_openorca,openhermes_2.5_mistral_7b,dolphin_2.2.1_mistral_7b \
#     agent=link_click/lmql_dfs_back_reminder \
#     agent.fewshot=0,1,2,3,4,5\
#     run_name=mistral_finetunes_fewshot \
#     > mistral_finetunes_fewshot.log 2>&1

#same script but a for loop that runs the same src/run.py for each model


# python src/run.py \
#     --multirun model.device=cuda model.half_precision=True model.batch_size=1 model=llama_2_7b_chat_hf\
#     agent=prompt_variations/description_system_fewshot_multiturn,prompt_variations/description_system_fewshot_system,prompt_variations/description_user_fewshot_multiturn,prompt_variations/description_user_fewshot_user \
#     run_name=prompt_variations_runai \
#     > prompt_variations_runai.log 2>&1


