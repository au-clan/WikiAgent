import attr
from attr import define, field
from typing import List, Any
import asyncio
from model_utils import BaseMessage, ChatModel

@define
class BatchHandler:
    model: ChatModel = field(init=False)
    batch_size: int = field(default=4)
    queue: asyncio.Queue = field(factory=asyncio.Queue)
    responses: dict = field(factory=dict)

    async def send_prompt(self, agent_id: int, prompt: list[BaseMessage]) -> BaseMessage:
        """Receive prompt from an agent, add it to the batch, and get the model's output."""
        response_event = asyncio.Event()
        await self.queue.put((agent_id, prompt, response_event))

        await response_event.wait()
        return self.responses.pop(agent_id)

    async def process_batch(self):
        """Process prompts in batches."""
        while True:
            batch = []
            events = []
            for _ in range(self.batch_size):
                if self.queue.empty():
                    break
                agent_id, prompt, event = await self.queue.get()
                batch.append(prompt)
                events.append((agent_id, event))

            outputs = self.model(batch)

            for (agent_id, event), output in zip(events, outputs):
                self.responses[agent_id] = output
                event.set()

            await asyncio.sleep(0)  # give control back to the event loop
