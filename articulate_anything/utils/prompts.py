BASIC_SYSTEM_PROMPT = "You are a helpful assistant."


def recognize_parts_from_image(image_paths, object_name):
    """
    Recognize the parts of an object from a set of images.

    Prompt structure informed by Articulate-AnyMesh (Qiu et al., arXiv:2502.02590;
    github.com/UMass-Embodied-AGI/Articulate-Anymesh); all wording here is original.
    """
    user_prompt = f"""
You are analyzing the physical structure of a single object.
The user supplies {len(image_paths)} photographs of one articulated object, each captured from a different angle.
Identify the object's main movable parts and report them as a JSON object with this structure:

```json
{{
    "part_list": [
        {{"part_name": "name of the part", "description": "a brief description about the part, and how it moves"}},
        {{"part_name": "name of the part", "description": "a brief description about the part, and how it moves"}},
        ...
    ]
    "fixed_part_name": "<object_name>_base",
}}
```

Guidelines:
(1) Return only what is asked for; include no extra commentary.
(2) Ground every conclusion strictly in the supplied images; do not invent parts that are not visible.
(3) Collapse repeated parts of the same kind into a single entry. If four drawers are present, list one entry named "drawer".
(4) Name parts in the singular (e.g. "drawer", not "drawers").
(5) List only parts attached by a joint that actually moves; omit anything rigid or fixed.
(6) Treat all remaining, non-moving geometry as one part called "base", which must not appear in part_list.
(7) Ignore the foot pedal of a trash can — do not include it as a part.

USER INPUT:
Object: {object_name}
    """

    system_prompt = """
You are a careful assistant with a strong grasp of how articulated objects are constructed.
    """


    return user_prompt, system_prompt


def generate_articulation_tree_known_parts(object_name, part_list):

    # TODO: remove instructions about foot pedal.
    """
    Generate the articulation tree of an object from a set of images.

    Prompt structure informed by Articulate-AnyMesh (Qiu et al., arXiv:2502.02590;
    github.com/UMass-Embodied-AGI/Articulate-Anymesh); all wording here is original.
    """
    user_prompt = f"""
You are analyzing the physical structure of a single object.
The user names an object, lists its main parts, and provides images of it from several viewpoints.
Group the listed parts into links and assign a joint type to each link.

After completing your analysis, report your answer as a JSON object with this structure:

```json
{{
    "parts": [
        {{"part_name": "name of the recognized part"}},
        ...
    ],
    "links": [
        {{"link_name": "name of the link"}},
        ...
    ],
    "joints": [
        {{"joint_name": "name of the joint", "joint_type": "type of the joint", "parent_link": "name of the parent link", "child_link": "name of the child link"}},
        ...
    ]
}}
```

Worked example:
```json
{{
    "parts": [
        {{"part_name": "Backrest"}},
        {{"part_name": "Seat"}},
        {{"part_name": "Caster wheel"}},
        {{"part_name": "Armrest"}}
    ],
    "links": [
        {{"link_name": "Base_link"}},
        {{"link_name": "Seat_link"}},
        {{"link_name": "Backrest_link"}},
        {{"link_name": "Caster wheel_link"}}
    ],
    "joints": [
        {{"joint_name": "caster_base_joint", "joint_type": "continuous", "parent_link": "Base_link", "child_link": "Caster wheel_link"}},
        {{"joint_name": "backrest_seat_joint", "joint_type": "revolute", "parent_link": "Seat_link", "child_link": "Backrest_link"}},
        {{"joint_name": "seat_base_joint", "joint_type": "prismatic", "parent_link": "Base_link", "child_link": "Seat_link"}}
    ]
}}
```

Guidelines:
(1) Return only what is asked for.
(2) The permitted joint types are exactly: fixed, prismatic, revolute and continuous.
(3) Give joint_type as a single word drawn from that set — nothing else.
(4) Form each link_name as part_name_link (for a part "Door", the link is "Door_link").
(5) The part list names part types, not individual instances. When a type has several instances, give each a unique suffix and its own link — e.g. two doors become door_1 and door_2 (format: semantic_part_name_unique_id).
(6) Ignore the foot pedal of a trash can — do not include it as a part.



USER INPUT:
Object: {object_name}
Parts:
{part_list}
    """

    system_prompt = """
You are a careful assistant with a strong grasp of articulated-object structure and deep familiarity with the URDF format.
    """

    return user_prompt, system_prompt

def merge_parts(object_name, parts_list):
    #TODO: this can probably be improved by adding the articulation tree to the user prompt and pictures of the actual object.

    user_prompt = f"""
    The user will provide you with images of {object_name}.
    The object in the images is divided into smaller segments, each labelled with a segmentation mask and a number on the mask. For each semantic part of the object, please choose the segments that belong to the semantic part.
    The colors of the segments are consistent across the images.

    The semantic parts we care about are: {", ".join(parts_list)} .

    First give an analysis of the semantics of each segment, relevant to the semantic parts we care about:

    json format:
    {{
    "analysis": {{
        "image_0": {{"orientation": "orientation of the object in the image", "segment_analysis": "analysis of the segments in the image"}},
        "image_1": {{"orientation": "orientation of the object in the image", "segment_analysis": "analysis of the segments in the image"}},
        ...
    }},
    "part_segment_dict": {{
        "part_name_1": [segment IDs that belong to this part, separated by commas],
        "part_name_2": [segment IDs that belong to this part, separated by commas],
        ...
    }}
    }}

    Remember:
    (1) We only care about the listed parts, do not add anything else to the part list
    (2) If some of the listed semantic parts do not exist in the image, omit these semantic parts
    (3) If there are multiple instances of a semantic part in the image, use two semantic parts. For example, if a fridge has two doors, use semantic part door_1 and door_2 (the format is semantic_part_name_unique_id). Make sure the semantic part name is exactly as specified in the user prompt.
    (4) Make sure the segment corresponds to the semantic part. For smaller objects, the number on the segment should be on the part itself, not the background.
    (5) If the segment is not part of the semantic part, omit it. Be careful not to include the background in the part list.
    (6) For parts that have handles that are rigidly connected to the part such as a drawer handle, include the handle as part of the part.
    (7)Do not include the handle if it is not rigidly connected to the part and can move independently, such as a door handle.
    """
    system_prompt = """You are an expert computer vision assistant specializing in multi-view 3D part segmentation analysis.

        Your task is to analyze a series of images showing different views of the same object. The object is divided into colored segments, each labeled with a consistent number.

        You will be given multiple images and a final user prompt asking you to identify the segments that correspond to a list of semantic parts.

        You must follow these rules:
        1.  **Analyze ALL provided images** to build a complete understanding. A segment's identity might only be clear from one or two angles.
        2.  **Correlate segments across views.** The colors and numbers for each segment are consistent.
        3.  **Strictly output JSON.** Your entire response must be a single, valid JSON object and nothing else. Do not add any explanatory text, apologies, or markdown formatting like ```json ... ```.

        Your JSON output must follow this exact schema:

        {{
        "analysis": {{
            "image_0": {{"orientation": "A brief description of the object's orientation in this image", "segment_analysis": "Your reasoning about the visible segments in this specific image."}},
            "image_1": {{"orientation": "...", "segment_analysis": "..."}},
            "...": "..."
        }},
        "part_segment_dict": {{
            "part_name_1": [list of segment IDs],
            "part_name_2": [list of segment IDs],
            "...": "..."
        }}
        }}"""

    return user_prompt, system_prompt
