import base64
import logging
from io import BytesIO
from google import genai
from google.genai.types import RawReferenceImage, MaskReferenceImage, MaskReferenceConfig, EditImageConfig, Image, \
    Part, Content, GenerateContentConfig, SafetySetting
# import vertexai
# from vertexai.preview.vision_models import Image as VertexImage
# from vertexai.preview.vision_models import RawReferenceImage, MaskReferenceImage, ImageGenerationModel
from openai import OpenAI
from PIL import Image as PILImage
import torch

from articulate_anything.utils.remote_retry import (
    RemoteCallFailed,
    handle_remote_exception,
)
from articulate_anything.utils.utils import assert_valid_key
from articulate_anything.utils.prompt_utils import (
    CLAUDE_VERSIONS,
    gemini_safety_settings,
    resolve_gemini_auth,
    claude_block_from_path,
    claude_response_text,
    claude_stream_message,
    get_claude_caps,
    make_claude_client,
)


BASIC_SYSTEM_PROMPT = "You are a helpful assistant."


class VLM_API:
    """
    Class for interfacing with remote VLM APIs, e.g.: ChatGPT, Gemini, etc.
    """
    VERSIONS = None

    @staticmethod
    def encode_image(image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            bytes: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class Gemini(VLM_API):
    """
    Class for interfacing with supported Gemini models
    """
    # TODO: Is this actually accurate?
    RESOLUTIONS = {
        "1:1": (1024, 1024),
        "3:4": (864, 1184),
        "4:3": (1184, 864),
        "9:16": (736, 1408),
        "16:9": (1408, 736),
    }
    IMAGE_SHAPES = {res for res in RESOLUTIONS.values()}

    # TODO: we need a central location for all models
    VERSIONS = {
        "gemini-2.0-flash-preview-image-generation": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 8192,
        },
        "gemini-2.5-pro": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-2.5-flash-image-preview": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },

        "gemini-2.5-flash-image": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
        "gemini-3.1-pro-preview": {
            "modalities": ["TEXT"],
            "max_tokens": 65535,
        },
        "gemini-3-pro-image-preview": {
            "modalities": ["TEXT", "IMAGE"],
            "max_tokens": 32768,
        },
    }
    def __init__(
        self,
        project=None,
        location="global",
        model="gemini-2.5-pro",
        verbose=False,
        api_key=None,
        backend=None,
    ):
        """
        Args:
            project (None or str): Vertex project. Required for the "vertex" route only.
            location (str): Location to use when calling the Gemini client
            model (str): Gemini model to use. Must be one of self.VERSIONS
            api_key (None or str): Gemini Developer API key. Falls back to
                GEMINI_API_KEY / GOOGLE_API_KEY.
            backend (None or str): "api_key", "vertex", or "auto" (default).
                See prompt_utils.resolve_gemini_auth.
        """
        self.project = project
        self.verbose = verbose
        if self.verbose:
            print("="*100)
            print(f"USING PROJECT: {self.project}")
            print("="*100)
        self.location = location
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Gemini model")
        self.model = model
        client_kwargs, self.auth_route = resolve_gemini_auth(
            project=project, location=location, api_key=api_key, backend=backend,
        )
        if self.verbose:
            print(f"Gemini auth route: {self.auth_route}", flush=True)
        self.client = genai.Client(**client_kwargs)
        

    def __call__(
        self,
        prompt,
        system_prompt=BASIC_SYSTEM_PROMPT,
        image_paths=None,
        temperature=0,
        top_p=0,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """
        Calls the Gemini model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_paths (None or str or list of str): If specified, absolute path(s) corresponding to reference image(s)
                to use as part of the overall prompt
            temperature (float): Temperature of the model when querying. Lower values correspond to more deterministic
                outputs
            top_p (float): Determines the cumulative probability of top-p tokens to select from probabilistically.
                E.g.: If top_p=0.7 and tokens a, b, c have probabilities of 0.4, 0.3, 0.2 respectively, only tokens
                a and b will be sampled from
            seed (int): Random seed to use
            n_retries (int): Number of retries to attempt
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            None or list of google.genai.types.GenerateContentResponse: Stream of responses generated from Gemini
        """
        parts = [Part.from_text(text=prompt)]
        if image_paths is not None:
            image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
            msg1_images = []
            for image_path in image_paths:
                msg1_images.append(Part.from_bytes(
                    data=self.encode_image(image_path),
                    mime_type="image/png",
                ))
            parts = msg1_images + parts
        contents = [Content(
            role="user",
            parts=parts,
        )]

        generate_content_config = GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            system_instruction=system_prompt,
            max_output_tokens=self.VERSIONS[self.model]["max_tokens"],
            response_modalities=self.VERSIONS[self.model]["modalities"],
            safety_settings=gemini_safety_settings(
                SafetySetting, route=self.auth_route
            ),
        )

        result = None
        # Budget grows if we hit 429s; see remote_retry.handle_remote_exception.
        budget = n_retries
        i = 0
        while i < budget:
            if self.verbose:
                print(f"Querying Gemini [{self.model}]: attempt {i + 1} of {budget}...")
            _result = []
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if print_results:
                        print(chunk.text, end="")
                    _result.append(chunk)
                result = _result
                break
            except Exception as e:
                budget = handle_remote_exception(
                    e, attempt=i, n_retries=budget,
                    provider="Gemini", model=self.model,
                )
            i += 1

        print()
        return result

    def get_result_text(self, result):
        return "".join(res.text for res in result)

    def get_result_images(self, result):
        images = []
        for res in result:
            for part in res.candidates[0].content.parts:
                if part.inline_data:
                    # The image data is in base64 encoded format within part.inline_data.data
                    image_data = part.inline_data.data

                    # You can then process this data, for example, save it as an image file
                    # Decode the base64 data and open it with PIL (Pillow)
                    image = PILImage.open(BytesIO(image_data))
                    images.append(image)
        return images


class Claude(VLM_API):
    """
    Class for interfacing with Anthropic Claude models served on Vertex AI.

    Drop-in for `Gemini` in the text+image steps of the pipeline: same
    constructor arguments, same `__call__` signature, same `get_result_text`.

    Claude takes text, images and PDFs -- **not video**. Callers holding a
    video must sample frames first.

    Auth is gcloud Application Default Credentials (`gcloud auth
    application-default login`) plus a project, exactly like the Gemini client
    above -- no Anthropic API key is involved.
    """

    VERSIONS = CLAUDE_VERSIONS

    def __init__(
        self,
        project,
        location="global",
        model="claude-opus-5",
        verbose=False,
        backend="vertex",
        api_key=None,
    ):
        """
        Args:
            project (str): GCP project to bill/route the Vertex AI request
            location (str): Vertex AI region. "global" is the safe default for
                Claude; Gemini-only regions (e.g. us-west1) may not serve it.
            model (str): Claude model to use, e.g. "claude-opus-5". On Vertex
                the id carries no provider prefix.
            backend (str): "vertex" (gcloud ADC) or "anthropic" (direct API).
            api_key (str, optional): Anthropic API key, `backend="anthropic"`
                only. Falls back to $ANTHROPIC_API_KEY.
        """
        self.project = project
        self.location = location or "global"
        self.verbose = verbose
        self.backend = backend or "vertex"
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Claude model")
        self.model = model
        self.caps = get_claude_caps(model)
        if self.verbose:
            print("=" * 100)
            if self.backend == "vertex":
                print(f"USING PROJECT: {self.project} (Claude on Vertex AI, "
                      f"region={self.location})")
            else:
                print(f"USING the Anthropic API directly (Claude, "
                      f"backend={self.backend})")
            print("=" * 100)
        self.client = make_claude_client(api_key=api_key, cfg={
            "vlm_backend": self.backend,
            "gcloud_project": self.project,
            "claude_location": self.location,
        })

    def __call__(
        self,
        prompt,
        system_prompt=BASIC_SYSTEM_PROMPT,
        image_paths=None,
        temperature=0,
        top_p=0,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """
        Calls the Claude model through the Vertex AI client.

        Args:
            prompt (str): Text prompt to use
            system_prompt (str): System instruction
            image_paths (None or str or list of str): absolute path(s) to
                image (or PDF) files to include in the prompt
            temperature (float): only sent to models that accept sampling
                params. Opus 5 / Opus 4.8 / Sonnet 5 reject them, so the value
                is ignored there -- steer those with the prompt instead.
            top_p (float): ignored, see `temperature`
            seed (int): ignored; the Claude API has no seed parameter
            n_retries (int): Number of retries to attempt
            print_results (bool): Whether to print the response as it streams

        Returns:
            anthropic.types.Message: the completed response
        """
        content = []
        if image_paths is not None:
            image_paths = [image_paths] if isinstance(
                image_paths, str) else image_paths
            content.extend(claude_block_from_path(p) for p in image_paths)
        content.append({"type": "text", "text": prompt})

        kwargs = {
            "model": self.model,
            "max_tokens": self.caps["max_tokens"],
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        if self.caps["thinking"]:
            kwargs["thinking"] = {"type": self.caps["thinking"]}
        if self.caps["effort"]:
            kwargs["output_config"] = {"effort": "high"}
        if self.caps["sampling"]:
            kwargs["temperature"] = temperature

        last_error = None
        # Budget grows if we hit 429s; see remote_retry.handle_remote_exception.
        budget = n_retries
        i = 0
        while i < budget:
            if self.verbose:
                print(f"Querying Claude [{self.model}]: attempt {i + 1} of "
                      f"{budget}...")
            try:
                result = claude_stream_message(
                    self.client, kwargs, print_results=print_results)
                logging.info(f"Claude [{self.model}] usage: {result.usage}")
                print()
                return result
            except Exception as e:
                last_error = e
                budget = handle_remote_exception(
                    e, attempt=i, n_retries=budget,
                    provider="Claude", model=self.model,
                )
            i += 1

        raise RemoteCallFailed(
            f"Claude [{self.model}] failed after {budget} attempts"
        ) from last_error

    def get_result_text(self, result):
        return claude_response_text(result)


# class Imagen3(VLM_API):
#     """
#     Class for interfacing with supported Imagen3 models
#     """
#     VERSIONS = {
#         "imagen-3.0-capability-001",  # Imagen3
#         "imagen-3.0-generate-002",  # Imagen3 -- currently not allowed for our account )):
#         "imagen-3.0-generate-001",  # Imagen3
#         "imagegeneration@006",      # Imagen2
#         "imagegeneration@002",      # Imagen
#     }

#     RESOLUTIONS = {
#         "1:1": (1024, 1024),
#         "3:4": (896, 1280),
#         "4:3": (1280, 896),
#         "9:16": (768, 1408),
#         "16:9": (1408, 768),
#     }

#     def __init__(
#         self,
#         project,
#         location="us-central1",
#         model="imagen-3.0-capability-001",
#         verbose=False,
#     ):
#         """
#         Args:
#             project (str): Name of the project to use when calling the Gemini client
#             location (str): Location to use when calling the Gemini client
#             model (str): Gemini model to use. Must be one of self.VERSIONS
#         """
#         self.project = project
#         self.verbose = verbose
#         if self.verbose:
#             print("="*100)
#             print(f"vlm.py line 251: USING PROJECT: {self.project}")
#             print("="*100)
#         self.location = location
#         assert_valid_key(key=model, valid_keys=self.VERSIONS, name="Imagen3 model")
#         self.model = model
#         vertexai.init(project=project, location=location)
#         self.client = ImageGenerationModel.from_pretrained(self.model)

#     def __call__(
#         self,
#         prompt,
#         image_path,
#         negative_prompt="",
#         mask_image_path=None,
#         edit_mode="default",
#         aspect_ratio="1:1",
#         n_images=4,
#         seed=0,
#         n_retries=3,
#         print_results=False,
#     ):
#         """
#         Calls the Imagen3 model using the client API.

#         Args:
#             prompt (str): Text prompt to use
#             image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
#                 the overall prompt
#             negative_prompt (str): Text prompt to use for negative prompting
#             mask_image_path (None or str): If specified, absolute path corresponding to reference image to use as a
#                 mask conditioning agent, e.g. for directly supervised image editing
#             edit_mode (str): Mode for Imagen3 to operate in, e.g.: "default", "inpainting-remove", ...
#             aspect_ratio (str): Aspect ratio to use for generated photos
#             n_images (int): Number of images to generate. Can be between 1-4
#             seed (int): Random seed to use
#             n_retries (int): Number of retries to attempt
#             print_results (bool): Whether to print results as they're being streamed

#         Returns:
#             None or list of Results: Imagen3 raw generated results
#         """
#         assert_valid_key(key=aspect_ratio, valid_keys=self.RESOLUTIONS, name="Aspect ratio")
#         raw_ref_image = RawReferenceImage(image=VertexImage.load_from_file(location=image_path), reference_id=1)
#         ref_images = [raw_ref_image]
#         if mask_image_path is not None:
#             mask_ref_image = MaskReferenceImage(reference_id=1,
#                                                 image=VertexImage.load_from_file(location=mask_image_path),
#                                                 mask_mode="foreground")
#             ref_images.append(mask_ref_image)
#         result = None
#         for i in range(n_retries):
#             if result is not None:
#                 break
#             print(f"Querying Imagen3 [{self.model}]: {i + 1} of {n_retries}...")
#             try:
#                 result = self.client._generate_images(
#                     prompt=prompt,
#                     edit_mode=edit_mode,
#                     reference_images=ref_images,
#                     seed=seed,
#                     number_of_images=n_images,
#                     negative_prompt=negative_prompt,
#                     aspect_ratio=aspect_ratio,
#                     safety_filter_level="block_few",
#                     person_generation="allow_adult",
#                 )

#             except:
#                 print(f"Failed attempt {i + 1} of {n_retries}")

#         if print_results and result is not None:
#             for res in result:
#                 res._pil_image.show()

#         return result

#     def get_result_images(self, result):
#         return [res._pil_image for res in result]



class GPT(VLM_API):
    """
    Class for interfacing with supported GPT models
    """
    VERSIONS = {
        "gpt-image-1",
    }

    IMAGE_SHAPES = {
        "portrait": "1024x1536",
        "square": "1024x1024",
        "landscape": "1536x1024",
    }

    def __init__(
        self,
        model="gpt-image-1",
        api_key=None,
    ):
        """
        Args:
            model (str): GPT model to use. Must be one of self.VERSIONS
            api_key (None or str): OpenAI API key to use. If not set, the OpenAI client falls back
                to the OPENAI_API_KEY environment variable
        """
        assert_valid_key(key=model, valid_keys=self.VERSIONS, name="GPT model")
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def __call__(
        self,
        prompt,
        image_path=None,
        n_images=1,
        n_retries=3,
        image_shape="square",
        print_results=False,
    ):
        """
        Calls the Gemini model using the client API.

        Args:
            prompt (str): Text prompt to use
            image_path (None or str): If specified, absolute path corresponding to reference image to use as part of
                the overall prompt
            n_images (int): Number of images to generate
            n_retries (int): Number of retries to attempt
            image_shape (str): Shape of the images to generate. Valid options: {portrait, square, landscape}
            print_results (bool): Whether to print results as they're being streamed

        Returns:
            dict: Output of the model. Valid keys are {"image", "text"} based on the desired model used
        """
        if self.model == "gpt-image-1":
            assert_valid_key(key=image_shape, valid_keys=self.IMAGE_SHAPES, name="image shape")

            result = None
            # Budget grows if we hit 429s; see remote_retry.handle_remote_exception.
            budget = n_retries
            i = 0
            while i < budget:
                print(f"Querying GPT [{self.model}]: {i + 1} of {budget}...")
                try:
                    result = self.client.images.edit(
                        model=self.model,
                        image=open(image_path, "rb"),
                        prompt=prompt,
                        n=n_images,
                        size=self.IMAGE_SHAPES[image_shape],
                        quality="high",
                        input_fidelity="high",
                        output_format="png",
                        background="auto",
                        # moderation="auto",
                    )
                    break
                except Exception as e:
                    budget = handle_remote_exception(
                        e, attempt=i, n_retries=budget,
                        provider="GPT", model=self.model,
                    )
                i += 1

            if print_results and result is not None:
                for dat in result.data:
                    image_base64 = dat.b64_json
                    image_bytes = base64.b64decode(image_base64)
                    PILImage.open(BytesIO(image_bytes)).show()

            return result

        else:
            raise ValueError(f"Got invalid GPT model for inference: {self.model}")

    def get_result_text(self, result):
        # return "".join(res.text for res in result)
        raise NotImplementedError

    def get_result_images(self, result):
        return [PILImage.open(BytesIO(base64.b64decode(dat.b64_json))) for dat in result.data]


