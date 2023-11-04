
#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status

source ~/.conda/etc/profile.d/conda.sh #|| { echo "Failed to source conda" > test.log; exit 1; }
conda activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

python src/run.py \
    --multirun model.device=cuda model.half_precision=True model.batch_size=2 \
    model.temperature=0.1,0.25,0.5,0.7,1,1.5 \
    agent=link_click/lmql_dfs_back \
    agent.fewshot=0,1,3,5,6 \
    run_name=constr_temp_fewshot \
    > constr_temp_fewshot.log 2>&1

# python src/run.py \
#     --multirun model.device=cuda model.half_precision=True model.batch_size=1 model=llama_2_7b_chat_hf\
#     agent=prompt_variations/description_system_fewshot_multiturn,prompt_variations/description_system_fewshot_system,prompt_variations/description_user_fewshot_multiturn,prompt_variations/description_user_fewshot_user \
#     run_name=prompt_variations_runai \
#     > prompt_variations_runai.log 2>&1


