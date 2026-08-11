from dataclasses import dataclass
from typing import Optional
from articulate_anything.utils.utils import (
    create_dir,
    load_json,
    join_path,
    file_to_string,
)
from articulate_anything.utils.prompt_utils import (
    setup_vlm_model,
    save_prompt_parts_as_html,
    is_claude_model,
)
from omegaconf import OmegaConf
import os
import logging
import tempfile
from IPython.display import display, HTML
from google.genai import types

GEN_CONFIG = {
    "temperature": 0.5,
}

# Claude ignores `temperature` (Opus 5 and Sonnet 5 reject sampling params
# outright) and instead takes `effort`, which trades thinking depth against
# cost. `max_tokens` covers thinking *and* the answer, so it needs headroom:
# the joint actor emits a full Python function.
GEN_CONFIG_CLAUDE = {
    "max_tokens": 32000,
    "effort": "high",
}

GEN_CONFIG_SIMFOUNDRY = types.GenerateContentConfig(
    temperature=0.5,
    candidate_count=3,
    thinkingConfig=types.ThinkingConfig(
        include_thoughts=True,
        thinking_level="high"
    )
)

@dataclass
class AgentConfig:
    out_dir: str
    api_key: Optional[str] = None
    n_retries: int = 3

class Agent:
    def __init__(
        self, cfg: AgentConfig,
    ):
        self.cfg = cfg
        self.n_retries = cfg.n_retries
        create_dir(cfg.out_dir)
        self.system_instruction = self.make_system_instruction()
        self.model = setup_vlm_model(
            model_name=cfg.model_name, system_instruction=self.system_instruction, api_key=cfg.api_key, cfg=cfg
        ) # could be model or client
        self.genai_model = "gemini" in cfg.model_name
        self.claude_model = is_claude_model(cfg.model_name)
        OmegaConf.save(self.cfg, join_path(cfg.out_dir, "config.json"))

    @ property
    def out_path(self):
        return join_path(self.cfg.out_dir, self.OUT_RESULT_PATH)

    @ property
    def error_path(self):
        return join_path(self.cfg.out_dir, "error.txt")

    def make_system_instruction(self):
        system_instruction = self._make_system_instruction()
        save_prompt_parts_as_html(
            system_instruction, join_path(
                self.cfg.out_dir, "system_instruction.html")
        )
        return system_instruction

    def load_system_instruction(self):
        return display(HTML(join_path(self.cfg.out_dir, "system_instruction.json")))

    def load_prompt_parts(self):
        return display(HTML(join_path(self.cfg.out_dir, "prompt.html")))

    def make_prompt_parts(self, *args, **kwargs):
        prompt_parts = self._make_prompt_parts(*args, **kwargs)
        save_prompt_parts_as_html(
            prompt_parts, join_path(self.cfg.out_dir, "prompt.html")
        )
        return prompt_parts

    def parse_response(self, response):
        raise NotImplementedError

    def _make_system_instruction(self):
        raise NotImplementedError

    def _make_prompt_parts(self, *args, **kwargs):
        raise NotImplementedError

    def _generate_content(self, prompt_parts, gen_config):
        """Call the configured SDK through the same path on initial and retry attempts."""

        if self.genai_model:
            contents = (
                types.Content(parts=prompt_parts)
                if self.__class__.__name__ == "JointCriticSimfoundryVideo"
                else prompt_parts
            )
            return self.model.models.generate_content(
                model=self.cfg.model_name,
                contents=contents,
                config=gen_config,
            )
        return self.model.generate_content(
            prompt_parts,
            generation_config=gen_config,
        )

    @staticmethod
    def _atomic_write_text(path, content):
        """Replace a small audit artifact atomically."""

        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def generate_prediction(self, *args, gen_config=None, overwrite=False, **kwargs):
        out_path = join_path(self.cfg.out_dir, self.OUT_RESULT_PATH)
        if (
            os.path.exists(out_path)
            and not overwrite
        ):
            logging.info(
                f"{self.__class__.__name__}: Prediction already exists at {out_path}. Skipping generation."
            )
            return

        if gen_config is None:
            if self.genai_model:
                gen_config = types.GenerateContentConfig(
                                    temperature=0.5,
                                    # candidate_count=3,
                                    system_instruction=self.system_instruction,
                                    # thinkingConfig=types.ThinkingConfig(
                                    #     include_thoughts=True,
                                    #     thinking_level="high"
                                    # )
                                )
            elif self.claude_model:
                gen_config = GEN_CONFIG_CLAUDE
            else:
                gen_config = GEN_CONFIG

        logging.info(f"{self.__class__.__name__}: Generating content.")
        prompt_parts = self.make_prompt_parts(*args, **kwargs)
        logging.info(f"Prompt: {prompt_parts}")



        logging.info(f"Generating content with model: {self.cfg.model_name}")
        response = self._generate_content(prompt_parts, gen_config)
     
        # logging.info(f"Usage: {response.usage_metadata}")

        for i in range(self.n_retries):
            try:
                self.parse_response(response, **kwargs)
                break
            except Exception as e:
                logging.error(f"Error parsing response: {e}")
                if i + 1 < self.n_retries:
                    response = self._generate_content(prompt_parts, gen_config)
        else:
            logging.error(f"Failed to parse response after {self.n_retries} retries")
            raise Exception(f"Failed to parse response after {self.n_retries} retries")

        # self.parse_response(response, **kwargs)
        # return response

    def load_prediction(self):
        if ".json" in self.OUT_RESULT_PATH:
            return load_json(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))
        return file_to_string(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))
