import logging
import random
import astor
import ast
import PIL
import re
import os
import json
# import google.generativeai as genai
from google import genai
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from articulate_anything.utils.utils import string_to_file
import textwrap
import markdown2
from io import BytesIO
import base64
from PIL import Image
import anthropic
from openai import OpenAI
 

GPT_VERSIONS = {
        "gpt-image-1",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
}


GEMINI_API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def resolve_gemini_auth(project=None, location="global", api_key=None, backend=None):
    """Pick the Gemini auth route. Returns (client_kwargs, route).

    route "api_key" -> Gemini Developer API (generativelanguage.googleapis.com);
    route "vertex"  -> Vertex AI (aiplatform.googleapis.com) via gcloud ADC.

    An API key and ADC are NOT interchangeable: a key identifies a project and
    carries no IAM principal, so Vertex cannot accept one. They are separate
    endpoints with separate quota and billing.

    backend forces a route ("api_key" or "vertex"); "auto" (the default) takes a
    key if one is configured and otherwise falls back to ADC. Override without
    touching code via SIMFOUNDRY_GEMINI_BACKEND.

    Kept byte-for-byte equivalent to simfoundry/models/vlm.py's copy so the two
    halves of the pipeline cannot drift apart on auth.
    """
    backend = (backend or os.environ.get("SIMFOUNDRY_GEMINI_BACKEND") or "auto").lower()
    if backend not in ("auto", "api_key", "vertex"):
        raise ValueError(
            f"Unknown Gemini backend {backend!r}: expected 'auto', 'api_key' or 'vertex'."
        )
    key = api_key or next(
        (os.environ[k] for k in GEMINI_API_KEY_ENVS if os.environ.get(k)), None
    )

    if backend == "vertex" or (backend == "auto" and not key):
        if not project:
            raise ValueError(
                "Gemini needs a Vertex project: pass project=, set GCLOUD_PROJECT, or "
                "supply an API key via api_key=/GEMINI_API_KEY. Vertex also needs ADC "
                "(`gcloud auth application-default login`, or a service-account JSON in "
                "GOOGLE_APPLICATION_CREDENTIALS)."
            )
        return {"vertexai": True, "project": project, "location": location}, "vertex"

    if not key:
        raise ValueError(
            "Gemini backend 'api_key' requires a key: pass api_key= or set "
            + " / ".join(GEMINI_API_KEY_ENVS) + "."
        )
    return {"api_key": key}, "api_key"


GEMINI_TEXT_HARM_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_HARASSMENT",
)
GEMINI_IMAGE_HARM_CATEGORIES = (
    "HARM_CATEGORY_IMAGE_HATE",
    "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
    "HARM_CATEGORY_IMAGE_HARASSMENT",
    "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
)


def gemini_safety_settings(SafetySetting, route="vertex", threshold="OFF"):
    """Safety settings for the given auth route.

    The Developer API rejects the HARM_CATEGORY_IMAGE_* categories with a 400
    INVALID_ARGUMENT, so they are sent on the vertex route only.
    """
    categories = GEMINI_TEXT_HARM_CATEGORIES
    if route != "api_key":
        categories += GEMINI_IMAGE_HARM_CATEGORIES
    return [SafetySetting(category=c, threshold=threshold) for c in categories]


def _cfg_attr(cfg, name, default=None):
    if cfg is None:
        return default
    try:
        value = cfg[name] if name in cfg else getattr(cfg, name, default)
    except Exception:
        value = getattr(cfg, name, default)
    return default if value is None else value


def setup_gemini(model_name, system_instruction=None, api_key=None, cfg=None):
    """Build the agent's Gemini client, honouring an API key or Vertex ADC.

    Returns None when neither is configured -- callers treat that as "no model".
    """
    try:
        client_kwargs, _route = resolve_gemini_auth(
            project=_cfg_attr(cfg, "gcloud_project"),
            location=_cfg_attr(cfg, "gcloud_location", "global"),
            api_key=api_key,
            backend=_cfg_attr(cfg, "gemini_backend"),
        )
    except ValueError:
        return None
    return genai.Client(**client_kwargs)

def setup_vlm_model(model_name, system_instruction=None, api_key=None, cfg=None):
    if model_name in GPT_VERSIONS:
        return setup_gpt(model_name, system_instruction, api_key)
    elif "gemini" in model_name:
        return setup_gemini(model_name, system_instruction, api_key, cfg)
    elif is_claude_model(model_name):
        return setup_claude(model_name, system_instruction, api_key, cfg)
    else:
        raise ValueError("Model name must contain 'gpt' or 'gemini' or 'claude' or 'o3'. Got: {}".format(model_name))

def setup_gemini_vertexai(model_name, system_instruction=None, api_key=None, cfg=None):
    # api_key = api_key or os.environ.get("API_KEY")
    # if not api_key:
    #     return None

    vertexai.init(project= cfg.gcloud_project, location=cfg.gcloud_location)
    # genai.configure(credentials=vertexai.auth.get_credentials())

    return GenerativeModel(model_name, system_instruction=system_instruction)
def assert_valid_key(key, valid_keys, name=None):
    """
    Helper function that asserts that @key is in dictionary @valid_keys keys. If not, it will raise an error.

    Args:
        key (any): key to check for in dictionary @dic's keys
        valid_keys (Iterable): contains keys should be checked with @key
        name (str or None): if specified, is the name associated with the key that will be printed out if the
            key is not found. If None, default is "value"
    """
    if name is None:
        name = "value"
    assert key in valid_keys, "Invalid {} received! Valid options are: {}, got: {}".format(
        name, valid_keys.keys() if isinstance(valid_keys, dict) else valid_keys, key
    )


class VLM_API:
    """
    Class for interfacing with remote VLM APIs, e.g.: ChatGPT, Gemini, etc.
    """
    VERSIONS = None

    @staticmethod
    def _encode_image_to_base64_from_path(image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            bytes: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def _encode_image_to_base64(pil_image):
        """
        Encodes PIL Image so that it can be included as part of GPT prompts
        """
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str



# ======================================================================
# Claude (Anthropic)
# ======================================================================
# Claude is reachable two ways, selected by `cfg.vlm_backend`:
#
#   vertex    -> Vertex AI (`AnthropicVertex`). Authenticates with gcloud
#                Application Default Credentials + `cfg.gcloud_project`, the
#                same auth the Gemini path already uses. This is the default.
#   anthropic -> the direct Anthropic API, which needs an API key.
#
# Claude accepts **text, images and PDFs -- but not video**. Callers that want
# to show Claude a video must sample frames first; use `supports_video_input`
# to branch (see JointCriticSimfoundryVideo).

CLAUDE_VERSIONS = {
    # model id -> request capabilities.
    #   thinking: value for the `thinking` request param (None -> omit it)
    #   effort:   whether `output_config.effort` is accepted
    #   sampling: whether temperature/top_p are accepted. Opus 5, Opus 4.8 and
    #             Sonnet 5 reject them with a 400, so we must not send them.
    "claude-opus-5":    {"max_tokens": 64000, "thinking": "adaptive", "effort": True,  "sampling": False},
    "claude-opus-4-8":  {"max_tokens": 64000, "thinking": "adaptive", "effort": True,  "sampling": False},
    "claude-sonnet-5":  {"max_tokens": 64000, "thinking": "adaptive", "effort": True,  "sampling": False},
    "claude-haiku-4-5": {"max_tokens": 32000, "thinking": None,       "effort": False, "sampling": True},
}

# For a Claude model we don't have an entry for, send only what every model
# accepts so an unknown id degrades instead of 400-ing.
CLAUDE_DEFAULT_CAPS = {"max_tokens": 16000, "thinking": None, "effort": False, "sampling": False}

CLAUDE_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def is_claude_model(model_name: str) -> bool:
    return "claude" in model_name


def supports_video_input(model_name: str) -> bool:
    """
    Whether the model accepts raw video as an input part.

    Only Gemini does. Claude and GPT are image/text only, so a caller holding a
    video must sample frames (`get_frames_from_video`) and send those instead.
    """
    return "gemini" in model_name


def get_claude_caps(model_name: str) -> dict:
    return CLAUDE_VERSIONS.get(model_name, CLAUDE_DEFAULT_CAPS)


def _cfg_get(cfg, key, default=None):
    """Read a key off a DictConfig/namespace/dict, tolerating missing keys."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    value = getattr(cfg, key, default)
    return default if value is None else value


def _claude_media_block(data: bytes, media_type: str):
    encoded = base64.b64encode(data).decode("utf-8")
    if media_type == "application/pdf":
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": encoded},
        }
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": encoded},
    }


def claude_block_from_path(path: os.PathLike):
    """Build an image or PDF content block from a file on disk."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".pdf":
        media_type = "application/pdf"
    elif ext in CLAUDE_IMAGE_MEDIA_TYPES:
        media_type = CLAUDE_IMAGE_MEDIA_TYPES[ext]
    else:
        raise ValueError(
            f"Claude cannot take '{ext}' as input ({path}). Supported: "
            f"{sorted(CLAUDE_IMAGE_MEDIA_TYPES)} and .pdf. "
            "Video must be sampled into frames first."
        )
    with open(path, "rb") as f:
        return _claude_media_block(f.read(), media_type)


def claude_block_from_pil(pil_image: Image.Image):
    """Build an image content block from a PIL image.

    Encoded as PNG: lossless, and unlike JPEG it round-trips RGBA renders.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return _claude_media_block(buffered.getvalue(), "image/png")


def claude_content_blocks(prompt_parts, model_name: str = ""):
    """
    Normalise the pipeline's heterogeneous prompt parts into Claude content
    blocks.

    Accepts the shapes the agents actually produce: plain strings, PIL images,
    already-formed Claude blocks (dicts), and `google.genai.types.Part` objects
    (used by the SIMFOUNDRY video critic).
    """
    if not isinstance(prompt_parts, (list, tuple)):
        prompt_parts = [prompt_parts]

    blocks = []
    for part in prompt_parts:
        if isinstance(part, str):
            if part:  # the API rejects empty text blocks
                blocks.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            blocks.append(claude_block_from_pil(part))
        elif isinstance(part, dict):
            blocks.append(part)
        elif getattr(part, "text", None) is not None:
            # google.genai.types.Part carrying text
            blocks.append({"type": "text", "text": part.text})
        elif getattr(part, "inline_data", None) is not None:
            # google.genai.types.Part carrying bytes
            inline_data = part.inline_data
            mime_type = getattr(inline_data, "mime_type", "") or ""
            data = getattr(inline_data, "data", None)
            if data is None:
                continue
            if mime_type.startswith("video/"):
                raise ValueError(
                    f"{model_name or 'Claude'} cannot take video as input "
                    f"(got a '{mime_type}' part). Sample frames with "
                    "`get_frames_from_video` and pass those images instead."
                )
            if not (mime_type.startswith("image/") or mime_type == "application/pdf"):
                raise ValueError(
                    f"{model_name or 'Claude'} cannot take '{mime_type}' as input. "
                    "Supported: image/*, application/pdf and text."
                )
            blocks.append(_claude_media_block(bytes(data), mime_type))
        else:
            raise TypeError(
                f"Unsupported prompt part for Claude: {type(part).__name__}"
            )
    return blocks


def claude_response_text(message) -> str:
    """
    Pull the assistant text out of a Claude response.

    Not simply `content[0].text`: with adaptive thinking on (the default on
    Opus 5) the first block is a thinking block, so we join every text block.
    """
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise RuntimeError(
            f"Claude declined the request (stop_reason=refusal, category={category}). "
            "Retry with a reworded prompt or fall back to another model."
        )

    text = "".join(
        block.text for block in message.content
        if getattr(block, "type", None) == "text"
    )

    if stop_reason == "max_tokens":
        logging.warning(
            "Claude hit max_tokens; the response is truncated. "
            "Raise `gen_config.max_tokens`."
        )
    if not text.strip():
        raise RuntimeError(
            f"Claude returned no text (stop_reason={stop_reason}). "
            "This usually means max_tokens was consumed by thinking."
        )
    return text


# Request params that only tune quality/cost. If an endpoint rejects one (a
# model or a Vertex region that doesn't support it), the sensible thing is to
# drop it and still get an answer rather than fail the pipeline step.
CLAUDE_OPTIONAL_PARAMS = ("output_config", "thinking")


def claude_stream_message(client, request_kwargs, print_results=False):
    """
    Run a streaming Claude request and return the completed message.

    Streaming (rather than a plain create) because `max_tokens` here is well
    above the point at which a non-streaming request risks an HTTP timeout.

    Retries once without the optional tuning params if the endpoint rejects
    them, so an unsupported `thinking`/`effort` degrades instead of failing.
    """
    def _run(kwargs):
        with client.messages.stream(**kwargs) as stream:
            if print_results:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
            return stream.get_final_message()

    try:
        return _run(request_kwargs)
    except anthropic.BadRequestError as e:
        dropped = [p for p in CLAUDE_OPTIONAL_PARAMS
                   if p in request_kwargs and p in str(e)]
        if not dropped:
            raise
        logging.warning(
            f"Claude rejected {dropped} ({e}); retrying without them. "
            "Consider removing the model from CLAUDE_VERSIONS' capability list."
        )
        return _run({k: v for k, v in request_kwargs.items()
                     if k not in dropped})


def make_claude_client(api_key=None, cfg=None):
    """
    Build the Claude client for the backend selected by `cfg.vlm_backend`.

    vertex (default): `AnthropicVertex`, authenticated with gcloud ADC. Run
        `gcloud auth application-default login` once and set `gcloud_project`.
        `claude_location` overrides `gcloud_location` -- useful because Claude
        and Gemini are not served in the same set of Vertex regions ("global"
        is the safe choice for Claude).
    anthropic: the direct API, keyed by `api_key` / $ANTHROPIC_API_KEY.
    """
    backend = _cfg_get(cfg, "vlm_backend", "vertex")

    if backend == "vertex":
        from anthropic import AnthropicVertex  # lazy: needs google-auth

        project = _cfg_get(cfg, "gcloud_project")
        if not project:
            raise ValueError(
                "Claude on Vertex AI needs `gcloud_project` in the config "
                "(e.g. export GCLOUD_PROJECT=<your-gcp-project>)."
            )
        region = _cfg_get(cfg, "claude_location") or _cfg_get(
            cfg, "gcloud_location", "global")
        logging.info(
            f"Claude via Vertex AI (project={project}, region={region})")
        return AnthropicVertex(project_id=project, region=region)

    api_key = api_key or os.environ.get(
        "ANTHROPIC_API_KEY") or os.environ.get("API_KEY")
    logging.info("Claude via the Anthropic API")
    return anthropic.Anthropic(api_key=api_key)


class ClaudeResponse:
    """Mirrors the `.text` attribute the agents expect from a Gemini response."""

    def __init__(self, text, message=None):
        self.text = text
        self.message = message


class ClaudeWrapper:
    """
    Claude behind the same `generate_content(prompt_parts, generation_config)`
    interface the GPT wrapper exposes, so `Agent` can use it unchanged.
    """

    def __init__(self, model_name, system_instruction=None, api_key=None,
                 cfg=None, client=None):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.caps = get_claude_caps(model_name)
        self.client = client or make_claude_client(api_key=api_key, cfg=cfg)

    def _build_request(self, content, generation_config):
        generation_config = generation_config or {}
        kwargs = {
            "model": self.model_name,
            "max_tokens": int(generation_config.get(
                "max_tokens", self.caps["max_tokens"])),
            "messages": [{"role": "user", "content": content}],
        }
        if self.system_instruction:
            kwargs["system"] = self.system_instruction
        if self.caps["thinking"]:
            kwargs["thinking"] = {"type": self.caps["thinking"]}
        if self.caps["effort"]:
            kwargs["output_config"] = {
                "effort": generation_config.get("effort", "high")}
        # Opus 5 / Opus 4.8 / Sonnet 5 reject sampling params outright.
        if self.caps["sampling"] and "temperature" in generation_config:
            kwargs["temperature"] = generation_config["temperature"]
        return kwargs

    def generate_content(self, prompt_parts, generation_config=None):
        content = claude_content_blocks(prompt_parts, self.model_name)
        kwargs = self._build_request(content, generation_config)

        logging.info(f"Querying Claude [{self.model_name}]")
        message = claude_stream_message(self.client, kwargs)

        logging.info(f"Claude [{self.model_name}] usage: {message.usage}")
        return ClaudeResponse(claude_response_text(message), message=message)


def setup_claude(model_name, system_instruction=None, api_key=None, cfg=None):
    """
    Build a Claude client wrapper.

    Args:
        model_name (str): Claude model id, e.g. "claude-opus-5". On Vertex AI
            the id carries no provider prefix.
        system_instruction (str, optional): system prompt for the model
        api_key (str, optional): Anthropic API key (`vlm_backend: anthropic`
            only; the Vertex backend uses gcloud ADC instead)
        cfg (DictConfig, optional): supplies `vlm_backend`, `gcloud_project`
            and `gcloud_location` / `claude_location`

    Returns:
        ClaudeWrapper
    """
    return ClaudeWrapper(
        model_name=model_name,
        system_instruction=system_instruction,
        api_key=api_key,
        cfg=cfg,
    )


class GPT(VLM_API):
    """
    Class for interfacing with supported GPT models
    """
    VERSIONS = {
        "gpt-image-1",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
    }

    IMAGE_SHAPES = {
        "portrait": "1024x1536",
        "square": "1024x1024",
        "landscape": "1536x1024",
    }

    def __init__(
        self,
        model_name="gpt-4o", 
        system_instruction=None,
        api_key=None,
    ):
        """
        Args:
            model_name (str): GPT model to use. Must be one of self.VERSIONS
            api_key (None or str): OpenAI API key to use. If not set, the OpenAI client falls back
                to the OPENAI_API_KEY environment variable
        """
        assert_valid_key(key=model_name, valid_keys=self.VERSIONS, name="GPT model")
        assert system_instruction is not None, "System instruction is required"
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def _format_content(self, prompt_parts):
        """Format content for OpenAI API, handling both text and images"""
        if not isinstance(prompt_parts, list):
            return prompt_parts  # Return as is if it's just a string

        formatted_content = []
        
        for part in prompt_parts:
            if isinstance(part, str):
                formatted_content.append({"type": "text", "text": part})
            elif isinstance(part, Image.Image):
                # Convert PIL Image to base64 and format for OpenAI
                base64_image = self._encode_image_to_base64(part)
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }
                })
            elif isinstance(part, dict) and part.get("type") == "image_url":
                # If it's already in the correct format, pass it through
                formatted_content.append(part)
        
        return formatted_content

    def generate_content(self, prompt_parts, generation_config={}):
        temperature = generation_config.get("temperature", 0.5)
        
        # Format the content for OpenAI
        formatted_content = self._format_content(prompt_parts)
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature)
            
        # Format response to match Gemini's format
        response_text = completion.choices[0].message.content
        
        # Create a MockResponse class to mimic the structure expected by parse_response
        class MockResponse:
            def __init__(self, text):
                self.text = text
        return MockResponse(response_text)

    # def __call__(
    #     self,
    #     prompt,
    #     image_path=None,
    #     n_images=1,
    #     n_retries=3,
    #     image_shape="square",
    #     print_results=False,
    # ):
    #     """
    #     Calls the GPT model using the client API.

    #     Args:
    #         prompt (str): Text prompt to use
    #         image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
    #             the overall prompt
    #         n_images (int): Number of images to generate
    #         n_retries (int): Number of retries to attempt
    #         image_shape (str): Shape of the images to generate. Valid options: {portrait, square, landscape}
    #         print_results (bool): Whether to print results as they're being streamed

    #     Returns:
    #         dict: Output of the model. Valid keys are {"image", "text"} based on the desired model used
    #     """
    #     if self.model == "gpt-image-1":
    #         assert_valid_key(key=image_shape, valid_keys=self.IMAGE_SHAPES, name="image shape")

    #         result = None
    #         for i in range(n_retries):
    #             if result is not None:
    #                 break
    #             print(f"Querying GPT [{self.model}]: {i + 1} of {n_retries}...")
    #             try:
    #                 result = self.client.chat
    #                 result = self.client.images.edit(
    #                     model=self.model,
    #                     image=open(image_path, "rb"),
    #                     prompt=prompt,
    #                     n=n_images,
    #                     size=self.IMAGE_SHAPES[image_shape],
    #                     quality="high",
    #                     input_fidelity="high",
    #                     output_format="png",
    #                     background="auto",
    #                     # moderation="auto",
    #                 )
    #             except:
    #                 print(f"Failed attempt {i + 1} of {n_retries}")

    #         if print_results and result is not None:
    #             for dat in result.data:
    #                 image_base64 = dat.b64_json
    #                 image_bytes = base64.b64decode(image_base64)
    #                 PIL.Image.open(BytesIO(image_bytes)).show()

    #         return result

    #     else:
    #         raise ValueError(f"Got invalid GPT model for inference: {self.model}")

    def get_result_text(self, result):
        # return "".join(res.text for res in result)
        raise NotImplementedError

    def get_result_images(self, result):
        return [PIL.Image.open(BytesIO(base64.b64decode(dat.b64_json))) for dat in result.data]




# class GPTWrapper:
#     def __init__(self, model_name, system_instruction, api_key):
#         self.model_name = model_name
#         self.system_instruction = system_instruction
#         self.api_key = api_key
#         self.client = OpenAI(api_key=self.api_key)

#     def _encode_image_to_base64(self, pil_image):
#         """Convert PIL Image to base64 string"""
#         buffered = BytesIO()
#         pil_image.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
#         return img_str

#     def _format_content(self, prompt_parts):
#         """Format content for OpenAI API, handling both text and images"""
#         if not isinstance(prompt_parts, list):
#             return prompt_parts  # Return as is if it's just a string

#         formatted_content = []
        
#         for part in prompt_parts:
#             if isinstance(part, str):
#                 formatted_content.append(part)
#             elif isinstance(part, Image.Image):
#                 # Convert PIL Image to base64 and format for OpenAI
#                 base64_image = self._encode_image_to_base64(part)
#                 formatted_content.append({
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                         "detail": "low"
#                     }
#                 })
#             elif isinstance(part, dict) and part.get("type") == "image_url":
#                 # If it's already in the correct format, pass it through
#                 formatted_content.append(part)
        
#         return formatted_content

#     def generate_content(self, prompt_parts, generation_config={}):
#         temperature = generation_config.get("temperature", 0.5)
        
#         # Format the content for OpenAI
#         formatted_content = self._format_content(prompt_parts)
        
#         completion = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {"role": "system", "content": self.system_instruction},
#                 {"role": "user", "content": formatted_content},
#             ],
#             temperature=temperature)
            
#         # Format response to match Gemini's format
#         response_text = completion.choices[0].message.content
        
#         # Create a MockResponse class to mimic the structure expected by parse_response
#         class MockResponse:
#             def __init__(self, text):
#                 self.text = text
#         return MockResponse(response_text)

def setup_gpt(model_name, system_instruction=None, api_key=None):
    return GPT(model_name=model_name, system_instruction=system_instruction, api_key=api_key)


    # Old method
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None
    
    return GPTWrapper(model_name=model_name, 
                      system_instruction=system_instruction, 
                      api_key=api_key)
    




def save_prompt_parts_as_html(prompt_parts, html_file_path):
    html_content = prompt_parts_to_html(prompt_parts)
    string_to_file(html_content, html_file_path)


def prompt_parts_to_html(prompt_parts, max_image_width=300, max_image_height=300):
    if isinstance(prompt_parts, (str, Image.Image)):
        # If a single string or image is passed, convert to a list
        prompt_parts = [prompt_parts]

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        <style>
            .image-row {{
                display: flex;
                flex-wrap: nowrap;
                margin-bottom: 20px;
            }}
            .image-row img {{
                margin-right: 1px;
                max-width: {max_width}px;
                max-height: {max_height}px;
                height: auto;
            }}
        </style>
    </head>
    <body>
    """.format(
        max_width=max_image_width, max_height=max_image_height
    )

    image_row_open = False

    for part in prompt_parts:
        if isinstance(part, str):
            if image_row_open:
                html_content += "</div>"
                image_row_open = False
            formatted_text = to_markdown(part)
            html_content += f"<p>{markdown2.markdown(formatted_text, extras=['fenced-code-blocks', 'code-friendly'])}</p>"
        elif isinstance(part, Image.Image):
            if not image_row_open:
                html_content += '<div class="image-row">'
                image_row_open = True
            buffered = BytesIO()
            part.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            html_content += f'<img src="data:image/png;base64,{img_str}">'

    if image_row_open:
        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    return html_content


def to_markdown(text):
    text = text.replace("•", "  *")
    return textwrap.indent(text, "> ", predicate=lambda _: True)


def prompt_parts_to_html_simfoundry(prompt_parts, output_dir=None, max_image_width=400, max_image_height=400, max_video_width=600):
    """
    Convert prompt parts from JointCriticSimfoundryVideo._make_prompt_parts to HTML.
    
    Handles:
    - types.Part with text
    - types.Part.from_bytes with image data (inline_data with image mime_type)
    - types.Part with inline_data containing video (Blob with video/mp4)
    - Plain strings
    - PIL Images (for backward compatibility)
    
    Args:
        prompt_parts: List of types.Part objects, strings, or PIL Images
        output_dir: Directory to save video files (required for video playback)
        max_image_width: Maximum width for images
        max_image_height: Maximum height for images
        max_video_width: Maximum width for videos
        
    Returns:
        HTML string
    """
    from google.genai import types
    
    if not isinstance(prompt_parts, list):
        prompt_parts = [prompt_parts]
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .content-block {{
                background: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .image-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 20px;
            }}
            .image-row img {{
                max-width: {max_image_width}px;
                max-height: {max_image_height}px;
                height: auto;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            video {{
                max-width: {max_video_width}px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            pre {{
                background-color: #f8f8f8;
                border-radius: 4px;
                padding: 10px;
                overflow-x: auto;
            }}
            .label {{
                font-size: 12px;
                color: #666;
                margin-bottom: 5px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
    <h1>Prompt Visualization</h1>
    """.format(
        max_image_width=max_image_width, 
        max_image_height=max_image_height,
        max_video_width=max_video_width
    )
    
    image_row_open = False
    video_counter = 0
    
    for i, part in enumerate(prompt_parts):
        # Concrete types first. The duck-typed `hasattr(part, 'text')` check
        # below would otherwise swallow PIL images: a PNG opened from disk is a
        # PngImageFile, whose `.text` holds the PNG metadata *dict*, not a
        # string. Backends that can't take video (Claude) send sampled PIL
        # frames through here, so this path is live.
        if isinstance(part, str):
            if image_row_open:
                html_content += "</div></div>"
                image_row_open = False
            formatted_text = to_markdown(part)
            html_content += f'<div class="content-block"><p>{markdown2.markdown(formatted_text, extras=["fenced-code-blocks", "code-friendly"])}</p></div>'

        elif isinstance(part, Image.Image):
            if not image_row_open:
                html_content += '<div class="content-block"><div class="label">Image</div><div class="image-row">'
                image_row_open = True
            buffered = BytesIO()
            part.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            html_content += f'<img src="data:image/png;base64,{img_str}">'

        # Handle google.genai types.Part objects
        elif getattr(part, 'text', None) is not None:
            # Text part
            if image_row_open:
                html_content += "</div></div>"
                image_row_open = False

            text = part.text
            formatted_text = to_markdown(text)
            html_content += f'<div class="content-block"><p>{markdown2.markdown(formatted_text, extras=["fenced-code-blocks", "code-friendly"])}</p></div>'

        elif hasattr(part, 'inline_data') and part.inline_data is not None:
            inline_data = part.inline_data
            mime_type = getattr(inline_data, 'mime_type', '')
            data = getattr(inline_data, 'data', None)
            
            if data is None:
                continue
                
            if mime_type.startswith('image/'):
                # Image data
                if not image_row_open:
                    if image_row_open:
                        html_content += "</div>"
                    html_content += '<div class="content-block"><div class="label">Image</div><div class="image-row">'
                    image_row_open = True
                
                # Encode image bytes to base64
                if isinstance(data, bytes):
                    img_str = base64.b64encode(data).decode("utf-8")
                else:
                    img_str = base64.b64encode(bytes(data)).decode("utf-8")
                html_content += f'<img src="data:{mime_type};base64,{img_str}">'
            
            elif mime_type.startswith('video/'):
                # Video data - save as file for browser display
                if image_row_open:
                    html_content += "</div></div>"
                    image_row_open = False
                
                if isinstance(data, bytes):
                    video_bytes = data
                else:
                    video_bytes = bytes(data)
                
                # Determine video extension
                ext = 'mp4' if 'mp4' in mime_type else 'webm' if 'webm' in mime_type else 'video'
                
                # Save video to output_dir
                if output_dir:
                    video_filename = f"prompt_video_{video_counter}.{ext}"
                    video_path = os.path.join(output_dir, video_filename)
                    with open(video_path, 'wb') as f:
                        f.write(video_bytes)
                    video_src = video_filename  # Relative path
                    video_counter += 1
                    
                    html_content += f'''
                    <div class="content-block">
                        <div class="label">Video (saved as {video_filename})</div>
                        <video controls autoplay loop muted>
                            <source src="{video_src}" type="{mime_type}">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    '''
                else:
                    # Fallback to base64 (may not work for large videos)
                    video_str = base64.b64encode(video_bytes).decode("utf-8")
                    html_content += f'''
                    <div class="content-block">
                        <div class="label">Video (base64 embedded - may not play if large)</div>
                        <video controls autoplay loop muted>
                            <source src="data:{mime_type};base64,{video_str}" type="{mime_type}">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    '''

    if image_row_open:
        html_content += "</div></div>"
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content


def save_prompt_parts_as_html_simfoundry(prompt_parts, html_file_path):
    """
    Save SIMFOUNDRY prompt parts (with types.Part objects) to an HTML file.
    Videos are saved as separate files in the same directory for better browser compatibility.
    
    Args:
        prompt_parts: List of types.Part objects from JointCriticSimfoundryVideo._make_prompt_parts
        html_file_path: Path to save the HTML file
    """
    output_dir = os.path.dirname(html_file_path)
    html_content = prompt_parts_to_html_simfoundry(prompt_parts, output_dir=output_dir)
    string_to_file(html_content, html_file_path)


def extract_code_from_string(code_string):
    """Extracts the Python code found between triple backticks (```) in a string."""
    pattern = r"```python\n([\s\S]*?)```"
    matches = re.findall(pattern, code_string, re.DOTALL)

    if matches:
        # dedent BEFORE strip. The model often emits the block already indented
        # (it is writing a function body). `.strip()` alone removes the leading
        # whitespace of the first line only, leaving line 1 at column 0 and the
        # rest at column 4 -- `IndentationError: unexpected indent` on otherwise
        # valid code. textwrap.dedent removes the COMMON prefix, so relative
        # indentation inside the block is preserved.
        return textwrap.dedent(matches[0]).strip()
    else:
        return None  # Indicate no code found


def remove_lines_containing(content: str, keyword: str) -> str:
    """Removes lines containing a specific keyword from the content string.

    Args:
        content (str): The content string to process.
        keyword (str): The keyword to search for.

    Returns:
        str: The content string with lines containing the keyword removed.
    """
    lines = content.split("\n")
    filtered_lines = [line for line in lines if keyword not in line]
    return "\n".join(filtered_lines)


def categorize_nodes(python_code: str):
    tree = ast.parse(python_code)

    imports = []
    other_top_level = []
    functions = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        else:
            other_top_level.append(node)

    return imports, other_top_level, functions


def create_new_module(imports, other_top_level, selected_functions):
    module = ast.Module(body=imports + other_top_level +
                        selected_functions, type_ignores=[])
    return astor.to_source(module)


def select_random_functions(functions, num_examples):
    num_examples = min(num_examples, len(functions))
    logging.info(
        f"Selecting {num_examples} examples")
    return random.sample(functions, num_examples)


def get_n_examples_from_python_code(python_code, num_examples):
    if isinstance(num_examples, int) and num_examples >= 0:
        imports, other_top_level, functions = categorize_nodes(python_code)
        selected_functions = select_random_functions(functions, num_examples)
        python_code = create_new_module(
            imports, other_top_level, selected_functions)
    return python_code

def extract_json_from_response(response):
    # 1. Find the start of the JSON block ('```json')
    start_index = response.find('```json')
    
    # 2. Find the end of the JSON block ('```')

    # Add 7 to the start index to search after '```json\n'
    end_index = response.find('```', start_index + 7) 
    
    if start_index != -1 and end_index != -1:
        json_str = response[start_index + 7:end_index].strip()
    else:
        # Fallback: if fences aren't found, try to strip off potential initial/trailing text, 
        json_str = response.strip().strip('```json').strip('```').strip()

    return json.loads(json_str)
