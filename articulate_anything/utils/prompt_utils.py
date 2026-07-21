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


def setup_gemini(model_name, system_instruction=None, api_key=None, cfg=None):
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    # No API key: fall back to Vertex AI auth (project/location), matching the
    # client used elsewhere in the pipeline (see articulate_anything/utils/vlm.py).
    # This keeps the agent working when the rest of the pipeline authenticates
    # via gcloud_project instead of GEMINI_API_KEY.
    gcloud_project = getattr(cfg, "gcloud_project", None) if cfg is not None else None
    if gcloud_project:
        gcloud_location = getattr(cfg, "gcloud_location", None) or "global"
        return genai.Client(
            vertexai=True,
            project=gcloud_project,
            location=gcloud_location,
        )

    return None

def setup_vlm_model(model_name, system_instruction=None, api_key=None, cfg=None):
    if model_name in GPT_VERSIONS:
        return setup_gpt(model_name, system_instruction, api_key)
    elif "gemini" in model_name:
        return setup_gemini(model_name, system_instruction, api_key, cfg)
    elif "claude" in model_name:
        return setup_claude(model_name, system_instruction, api_key)
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



class ClaudeWrapper:
    def __init__(self, model_name, system_instruction, api_key):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _encode_image_to_base64(self, pil_image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    def _format_content(self, prompt_parts):
        """Format content for Claude API, handling both text and images"""
        if not isinstance(prompt_parts, list):
            return [{"type": "text", "text": prompt_parts}]
        
        formatted_content = []
        
        for part in prompt_parts:
            if isinstance(part, str):
                formatted_content.append({
                    "type": "text",
                    "text": part
                })
            elif isinstance(part, Image.Image):
                base64_image = self._encode_image_to_base64(part)
                formatted_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            elif isinstance(part, dict):
                formatted_content.append(part)
        
        return formatted_content
    
    def generate_content(self, prompt_parts, generation_config={}):
        temperature = generation_config.get("temperature", 0)
        max_tokens = generation_config.get("max_tokens", 1024)
        
        formatted_content = self._format_content(prompt_parts)
        
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.system_instruction,
            messages=[
                {
                    "role": "user",
                    "content": formatted_content
                }
            ]
        )
        print(">>> USAGE", message.usage)
        
        class MockResponse:
            def __init__(self, text):
                self.text = text
        
        return MockResponse(message.content[0].text)
    

def setup_claude(model_name, system_instruction=None, api_key=None):
    """
    Setup function for Claude models.
    
    Args:
        model_name (str): Name of the Claude model (e.g., "claude-3-opus-20240229")
        system_instruction (str, optional): System instruction for the model
        api_key (str, optional): Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY in environment
        
    Returns:
        ClaudeWrapper or None: Initialized wrapper if successful, None if no API key available
    """
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None
    
    return ClaudeWrapper(
        model_name=model_name,
        system_instruction=system_instruction,
        api_key=api_key
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
        # Handle types.Part objects
        if hasattr(part, 'text') and part.text is not None:
            # Text part
            if image_row_open:
                html_content += "</div>"
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
        
        # Handle plain strings (backward compatibility)
        elif isinstance(part, str):
            if image_row_open:
                html_content += "</div></div>"
                image_row_open = False
            formatted_text = to_markdown(part)
            html_content += f'<div class="content-block"><p>{markdown2.markdown(formatted_text, extras=["fenced-code-blocks", "code-friendly"])}</p></div>'
        
        # Handle PIL Images (backward compatibility)
        elif isinstance(part, Image.Image):
            if not image_row_open:
                html_content += '<div class="content-block"><div class="label">Image</div><div class="image-row">'
                image_row_open = True
            buffered = BytesIO()
            part.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            html_content += f'<img src="data:image/png;base64,{img_str}">'
    
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
        # Return the first match of the Python code block, removing whitespace
        return matches[0].strip()
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
