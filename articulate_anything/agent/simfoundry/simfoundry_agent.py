from articulate_anything.agent.agent import Agent


SIMFOUNDRY_SYSTEM_INSTRUCTION = """
You are a helpful assistant.
"""

class SimfoundryAgent(Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _make_system_instruction(self):
        return SIMFOUNDRY_SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, *args, **kwargs):
        return super()._make_prompt_parts(*args, **kwargs)