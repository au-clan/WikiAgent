import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import os
import pandas as pd
from pprint import pprint
from time import time
import asyncio

from model_utils import SystemMessage, HumanMessage, AIMessage, BaseMessage
from prompt_utils import get_navigate_entire_path_prompt
from logger import get_logger
from utils import printc, load_tsv_to_dict_list
from navigate_links_only import get_path_demonstrations, get_paths_real_sample
from environments import LinkWikiEnviroment
from agents import LinkClickAgent
from batch_handler import BatchHandler

os.environ['HYDRA_FULL_ERROR'] = '1'

log = get_logger()

@hydra.main(version_base = None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model = instantiate(cfg.model)
    log.debug("Model instantiated.")
    fewshot = 5
    paths_real = get_paths_real_sample(n=100)
    path_demonstrations = get_path_demonstrations()
    model_output_df = pd.DataFrame(columns=['src', 'tgt', 'human_path', 'model_output', 'history', 'temperature', 'top_p'])

    for path in paths_real:
        start = time()
        messages = get_navigate_entire_path_prompt(**cfg.experiments.prompts,
                                                   path_demonstrations=path_demonstrations,
                                                   src=path["src"], tgt=path["tgt"],
                                                   fewshot=fewshot, n=1)
        response = model(messages)

        model_output_df = pd.concat([model_output_df, pd.DataFrame({'src': path["src"], 'tgt': path["tgt"], 'human_path': path["output"],
                                                                    'model_output': response, 'history': model.get_history_str(),
                                                                    'temperature': model.temperature, 'top_p': model.top_p}, index=[0])], ignore_index=True)
        print(f"Time taken: {time() - start} seconds")
        print(response)
        break
        #model_output_df.to_csv(os.path.join(cfg.output_dir, f"{model.model_name.split('/')[-1]}_fewshot_{fewshot}_outputs.tsv"), index=False, sep='\t')

###
@hydra.main(version_base = None, config_path="configs", config_name="config")
async def test(cfg: DictConfig):
    env = LinkWikiEnviroment("Andy Warhol")
    model = instantiate(cfg.model)

    handler = BatchHandler(model=model)
    agents = [LinkClickAgent(handler=handler, agent_id=i, environment=env) for i in range(4)]
    
    agent_tasks = [asyncio.create_task(agent.act()) for agent in agents]
    batch_processor = asyncio.create_task(handler.process_batch())

    await asyncio.gather(*agent_tasks)
    batch_processor.cancel()


if __name__ == "__main__":
    #main()
    asyncio.run(test())
