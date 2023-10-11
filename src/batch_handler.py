import attr
from attr import define, field
from typing import List, Any
import asyncio
from model_utils import BaseMessage,AIMessage, ChatModel
from utils import printc

@define
class BatchHandler:
    """
    Handles batching and sending prompts to the model.
    TODO:
    - For a queue < batch_size, right now it doesn't send the prompt to the model.
    """
    model: ChatModel = field(default=None)
    batch_size: int = field(default=1)
    queue: asyncio.Queue = field(factory=asyncio.Queue)
    responses: dict = field(factory=dict)
    debug_mode: bool = field(default=False)

    async def send_prompt(self, agent_id: int, prompt: list[BaseMessage]) -> BaseMessage:
        """Receive prompt from an agent, add it to the batch, and get the model's output."""
        response_event = asyncio.Event()
        await self.queue.put((agent_id, prompt, response_event))
        printc(f"Prompt added to queue for agent {agent_id}.", c="y")
        await response_event.wait()
        printc(f"Response received for agent {agent_id}.", c="y")
        return self.responses.pop(agent_id)

    async def process_batch(self):
        """Process prompts in batches."""
        while True:
            if self.queue.empty():
                # if the loop restarts and the queue is still empty, there's probably no active agents
                break
            
            batch = []
            events = []

            for _ in range(self.batch_size):
                if self.queue.empty():
                    break
                agent_id, prompt, event = await self.queue.get()
                printc(f"Processing prompt for agent {agent_id}.", c="y")
                batch.append(prompt)
                events.append((agent_id, event))

            if self.debug_mode:
                outputs = [AIMessage(content='{"click": "United States"}') for _ in batch]
            else:
                printc("Sending batch to model...", c="y")
                outputs = self.model(batch)

            for (agent_id, event), output in zip(events, outputs):
                self.responses[agent_id] = output
                event.set()

            await asyncio.sleep(0.1)  # give control back to the event loop

            