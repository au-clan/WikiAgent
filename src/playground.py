# File to play around with the code

from environments import *
from agents import *
from model_utils import *
from batch_handler import *
from pprint import pprint

async def main():
    env = LinkWikiEnviroment("Andy Warhol")
    handler = BatchHandler()
    agents = [LinkClickAgent(handler=handler, agent_id=i, environment=env) for i in range(4)]
    
    agent_tasks = [asyncio.create_task(agent.act()) for agent in agents]


if __name__ == "__main__":
    asyncio.run(main())