
#!/bin/bash


#set -e # Exit immediately if a command exits with a non-zero status

source ~/.conda/etc/profile.d/conda.sh #|| { echo "Failed to source conda" > test.log; exit 1; }
conda activate ../miniconda3/envs/wikinav #|| { echo "Failed to activate conda environment" > test.log; exit 1; }

MODELS=("mistral_7b_instruct") 
for model in "${MODELS[@]}"; do
    python src/run.py \
        --multirun model.device=cuda model.dtype=bfloat16 model.batch_size=2 \
        model="${model}" \
        agent=link_click/lmql_simplified \
        agent.fewshot=0,1,2,3,4\
        run_name="${model}_simplified_constr_fewshot" \
        > "${model}_simplified_constr_fewshot.log" 2>&1
done

python src/run.py \
    --multirun model.device=cuda model.batch_size=2 \
    model=mistral_7b_openorca model.dtype=bfloat16 \
    agent=link_click/lmql_dfs_back_reminder_reason \
    run_name=link_click_reason \
    > link_click_reason.log 2>&1

# python src/run.py \
#     --multirun model.device=cuda model.half_precision=True model.batch_size=2 \
#     model=mistral_7b_instruct,mistral_7b_openorca,openhermes_2.5_mistral_7b,dolphin_2.2.1_mistral_7b \
#     agent=link_click/lmql_dfs_back_reminder \
#     agent.fewshot=0,1,2,3,4,5\
#     run_name=mistral_finetunes_fewshot \
#     > mistral_finetunes_fewshot.log 2>&1

#same script but a for loop that runs the same src/run.py for each model
# command to test batch size
# python src/run.py --multirun model.device=cuda model.batch_size=2 model.dtype=bfloat16 agent=link_click/lmql_dfs_back_reminder_reason run_name=link_click_reason experiment.override=True experiment.num_agents=5 agent.max_actions=1 agent.fewshot=0,1,2,3,4
