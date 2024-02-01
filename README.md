# WikiAgent
Master thesis project on navigating Wikipedia with an LLM-based agent.

## Reproducibility
For DLAB users, we recommend performing the following steps in ICCluster, in the `/dlabdata1/<your-username>` directory, as it is accessible from RunAI pods.

### Initial setup and data
- Clone this repo, and create an environment with the packages in `requirements.txt`.
- Download the data from [here](link). Unzip the files and place them in the `data` directory. This data consists of the `wikispeedia` (embeddings, wikispeedia missions, etc.) and `wpcd` (Wikipedia articles) folders.

### Prepare the config
We use the `hydra-core` library to manage configurations of agents, models, experiments, etc., in the `src/configs` directory. 
- In `src/configs/config.yaml` is the main config file. Change the directories based on your setup. For DLAB users, you can just change your username in `workdir`.

### Quick start
The following command runs a single experiment with the ReAct constrained WikiAgent on 100 missions sampled randomly from the Wikispeedia user missions. The main output of this experiment will be the file with all the agent actions in `logs/multiruns/test_run/0/agent_runs.tsv`.

```bash
python src/run.py --multirun model.batch_size=2\
    experiment=random \
    model=mistral_7b_instruct_v0.2 \
    agent=navigate/lmql_headings_reason \
    experiment.num_agents=100 \
    +agent.verbose=True \
    run_name="test_run"
```
Here's the breakdown of the command:
- `python src/run.py`: The main script to run an experiment.
- `--multirun`: Hydra command for running multiple experiments in a row. In this case, we are running a single experiment.
- `model.batch_size=2`: The batch size of the model.
- `experiment=random`: The experiment to run.
    - `random`: randomly samples unique missions from the Wikispeedia user missions.
    - `human_test_set`: samples missions that have been played at least 4 times by humans.
- `model=mistral_7b_instruct_v0.2`: The model to use. The models are defined in `src/config/model`. To create a new model, just copy one of the existing models and change the parameters.
- `agent=navigate/lmql_headings_reason`: This is the ReAct constrained WikiAgent.
    -  `navigate` -> WikiAgent; `link_click` -> links-only WikiAgent.
    - `lmql`-> constrained decoding.
    - `reason` -> reasoning before action (ReAct).
- `experiment.num_agents=100`: The number of agents to run.
- `+agent.verbose=True`: Print the prompt at each step.
- `run_name="test_run"`: The name of the run. The logs are saved in `logs/multiruns/<run_name>`.


### Experiments
All scripts for the experiments are in the `scripts` directory. Before running them, make sure to change the folders to your setup. The naming convention for the experiments is mostly `<mission-sample>_<agent-type>_<index>.sh`. The `<index>` is used to split the same experiment into multiple parallel scripts.

For the `<mission-sample>`:
- `random`: randomly sampled unique missions from the Wikispeedia user missions.
- `human`: samples missions that have been played at least 4 times by humans.
- `click`: for click comparison experiments.

For the `<agent-type>`:
- `link`: links-only WikiAgent.
- `nav`: WikiAgent.

