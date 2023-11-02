
#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status

source ~/.conda/etc/profile.d/conda.sh #|| { echo "Failed to source conda" > test.log; exit 1; }
conda activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

# python src/run.py \
#     --multirun model.device=cuda model.half_precision=True model.batch_size=1 model=mistral_7b_instruct\
#     run_name=mistral_test \
#     > mistral_test.log 2>&1

python src/run.py \
    --multirun model.device=cuda model.half_precision=True model.batch_size=1 \
    model.temperature=0.2,0.7,0.5,1,1.5 \
    agent=link_click/dfs_back_reason \
    run_name=reason_temperatures \
    > reason_temperatures.log 2>&1

# python src/run.py \
#     --multirun model.device=cuda model.half_precision=True model.batch_size=1 model=llama_2_7b_chat_hf\
#     agent=prompt_variations/description_system_fewshot_multiturn,prompt_variations/description_system_fewshot_system,prompt_variations/description_user_fewshot_multiturn,prompt_variations/description_user_fewshot_user \
#     run_name=prompt_variations_runai \
#     > prompt_variations_runai.log 2>&1


