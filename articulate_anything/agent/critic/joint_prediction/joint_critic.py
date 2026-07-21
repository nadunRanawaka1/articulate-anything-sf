
import os
import glob
from articulate_anything.agent.multimodal_incontext_agent import InContextExampleModel
from articulate_anything.utils.viz import get_frames_from_video
from articulate_anything.utils.utils import (
    file_to_string,
    file_to_string_python_prediction,
    join_path,
    save_json,
    load_json,
    create_task_config,
)
import logging
from articulate_anything.agent.agent import Agent
import json
from articulate_anything.api.odio_urdf import get_semantic_joint_id
from PIL import Image
from google.genai import types
from articulate_anything.utils.prompt_utils import save_prompt_parts_as_html_simfoundry
CRITIC_INSTRUCTION = """
## General Instructions

You are a visual critic expert whose job is to assess the realism of a joint prediction of a 3D model.

You will analyze a candidate function `partnet_{object_id}`. Assess how realistic this model is compared to the ground truth.

You will see two videos: first the ground truth, then the prediction.

Compare these videos and provide feedback on the prediction. Use this format:

```json
{
"gt_description": {describe the gt video},
"pred_description": {describe the prediction video},
"candidate_function_description": {describe the candidate function},
"failure_reason": {one of these "success", "joint_type", "joint_axis", "joint_origin", "joint_limit"},
"improvement_suggestion": {suggestion to improve the prediction},
"realism_rating": {0-10},
}
```

Be concise and specific. When writing the description, compare the predicted video to the ground truth and analyze the `candidate_function` to identify issues.

Important points:

- Evaluate only the joint prediction, not link placement.
- Compare videos first, then examine the candidate function.
- Rate highly if the prediction closely matches the ground truth.
- Identify problems using this checklist, focusing on the most significant error:
  1. Incorrect joint type (e.g., revolute instead of prismatic): Rate 0
  2. Wrong joint axis (e.g., x-axis instead of y-axis): Rate 1
  3. Incorrect joint origin (for **revolute joints** only): Rate 2
  4. Incorrect joint limit (for **revolute joints** only; e.g, the door is opening inward instead of outward): Rate 3
  5. No errors: Rate above 5, mark as "success"
- Your `realism_rating` must match the `failure_reason` according to the ratings specified above.
- Joint axis order is [x, y, z]: 
    - x : forward -- positive x, backward -- negative x
    - y: right -- positive y, left -- negative y
    - z: up -- positive z, down -- negative z
- Use the `candidate_function` to confirm your diagnosis.
- Analyze the videos frame-by-frame if needed. Describe the motion clearly, using terms like "rotates", "slides", or "pivots" to convey the joint behavior.
- **Important**: the groundtruth video might not have the same texture as the prediction video e.g., the gt might be in-the-wild video captured by a phone while prediction is 3D model rendered in a 
physics simulator. Thus, you must correctly describe the motion of the object in the video and compare it with the prediction.
- We will use `json.loads()` to parse your response. Make sure that your response is exactly ```json {your response}```, nothing more, nothing less.
"""

CRITIC_INSTRUCTION_SIMFOUNDRY = """
## General Instructions

You are a visual critic expert whose job is to assess the realism of a joint prediction of a 3D model.

You will analyze a candidate function `partnet_{object_id}`. Assess how realistic this model is compared to the ground truth.

You will first see the ground truth image of the object, then a video of the prediction.

**CRITICAL: You MUST analyze the actual visual content in the video frames. DO NOT infer motion from the candidate function code or object type. ONLY describe what you actually SEE in the video frames.**

## Example Analysis Process:

1. **First, describe EXACTLY what you see in EACH video frame:**
   - Frame 1: What is the position/state of the moving part?
   - Frame 2: Has it moved? In what direction? How far?
   - Frame 3: Continuing the same motion, or different?
   - Frame 4-5: Final position/state?

2. **Then, determine the motion type from your visual observations:**
   - If the part moves in a STRAIGHT LINE → prismatic/sliding
   - If the part moves in an ARC/ROTATION → revolute/rotating
   - If NO MOVEMENT VISIBLE → static/broken

3. **Finally, compare with the candidate function:**
   - Does the code match what you ACTUALLY SAW?
   - Ignore what the code CLAIMS - focus on visual evidence

Compare the ground truth image and the prediction video and provide feedback. Use this format:

```json
{
"gt_description": {describe the gt image},
"frame_by_frame_analysis": {describe what you SEE in each frame: "Frame 1: X, Frame 2: Y, Frame 3: Z..."},
"observed_motion_type": {"sliding_linear" or "rotating_arc" or "no_visible_motion"},
"pred_description": {summarize the motion you observed, or "No video provided" if missing},
"candidate_function_description": {describe the candidate function},
"failure_reason": {one of these "success", "joint_type", "joint_axis", "joint_origin", "joint_limit", "no_prediction"},
"improvement_suggestion": {suggestion to improve the prediction},
"realism_rating": {0-10},
}
```

**CRITICAL RULES:**

- **LOOK AT THE VIDEO FIRST, CODE SECOND**: Base your evaluation ONLY on what you visually observe in the frames
- **BE SPECIFIC**: Reference actual visual details (colors, positions, parts) that prove you looked at the frames
- **DESCRIBE BEFORE JUDGING**: Complete frame_by_frame_analysis before determining failure_reason
- Evaluate only the joint prediction, not link placement.
- Rate highly ONLY if the prediction closely matches the ground truth AND the video shows the expected motion.
- Identify problems using this checklist, focusing on the most significant error:
  0. **No prediction video / Missing video / No visible motion**: Rate 0, mark as "no_prediction"
  1. Incorrect joint type (e.g., revolute instead of prismatic based on OBSERVED motion): Rate 0
  2. Wrong joint axis (e.g., x-axis instead of y-axis based on OBSERVED direction): Rate 1
  3. Incorrect joint origin (for revolute joints only, based on OBSERVED pivot point): Rate 2
  4. Incorrect joint limit (based on OBSERVED range/direction): Rate 3
  5. No errors: Rate above 5, mark as "success"
- Your `realism_rating` must match the `failure_reason` according to the ratings specified above.
- Joint axis order is [x, y, z]: 
    - x: forward -- positive x, backward -- negative x
    - y: right -- positive y, left -- negative y
    - z: up -- positive z, down -- negative z
- **Important**: the groundtruth image might not have the same texture as the prediction video e.g., the gt might be in-the-wild image while prediction is a 3D render.
- **Important**: we may use surface meshes for the object, in this case, the interior of the object may be hollow. Consider if the surface of the relevant part moves correctly.
- We will use `json.loads()` to parse your response. Make sure that your response is exactly ```json {your response}```, nothing more, nothing less.
"""

CRITIC_INSTRUCTION_SIMFOUNDRY_VIDEO = """
## General Instructions

You are a visual critic expert whose job is to assess the realism of a joint prediction of a 3D model.

You will analyze a candidate function `partnet_{object_id}`. Assess how realistic this model is compared to the ground truth.

You will first see the ground truth image of the object, then a video of the prediction.

**CRITICAL: You MUST analyze the actual visual content in the video frames. DO NOT infer motion from the candidate function code or object type. ONLY describe what you actually SEE in the video frames.**


Compare the ground truth image and the prediction video and provide feedback. Use this format:

```json
{
"gt_description": {describe the gt image},
"pred_description": {summarize the motion you observed, or "No video provided" if missing},
"observed_motion_type": {"sliding_linear" or "rotating_arc" or "no_visible_motion"},
"candidate_function_description": {describe the candidate function},
"failure_reason": {one of these "success", "joint_type", "joint_axis", "joint_origin", "joint_limit", "no_prediction"},
"improvement_suggestion": {suggestion to improve the prediction},
"realism_rating": {0-10},
}
```

**CRITICAL RULES:**

- **LOOK AT THE VIDEO FIRST, CODE SECOND**: Base your evaluation ONLY on what you visually observe in the frames
- **BE SPECIFIC**: Reference actual visual details (colors, positions, parts) that prove you looked at the frames
- **DESCRIBE BEFORE JUDGING**: Complete video analysis before determining failure_reason
- Evaluate only the joint prediction, not link placement.
- Rate highly ONLY if the prediction closely matches the ground truth AND the video shows the expected motion.
- Identify problems using this checklist, focusing on the most significant error:
  0. **No prediction video / Missing video / No visible motion**: Rate 0, mark as "no_prediction"
  1. Incorrect joint type (e.g., revolute instead of prismatic based on OBSERVED motion): Rate 0
  2. Wrong joint axis (e.g., x-axis instead of y-axis based on OBSERVED direction): Rate 1
  3. Incorrect joint origin (for revolute joints only, based on OBSERVED pivot point): Rate 2
  4. Incorrect joint limit (based on OBSERVED range/direction): Rate 3
  5. No errors: Rate above 5, mark as "success"
- Your `realism_rating` must match the `failure_reason` according to the ratings specified above.
- Joint axis order is [x, y, z]: 
    - x: forward -- positive x, backward -- negative x
    - y: right -- positive y, left -- negative y
    - z: up -- positive z, down -- negative z
- **Important**: the groundtruth image might not have the same texture as the prediction video e.g., the gt might be in-the-wild image while prediction is a 3D render.
- **Important**: we may use surface meshes for the object, in this case, the interior of the object may be hollow. Consider if the surface of the relevant part moves correctly.
- We will use `json.loads()` to parse your response. Make sure that your response is exactly ```json {your response}```, nothing more, nothing less.
"""



CRITIC_COTRACKER_TRACE = """
## CoTracker Motion Tracing

We also use a motion tracker algorithm (CoTracker) to highlight the moving parts in the videos. Pay close attention to the motion traces annotated in the videos to gain
information about the joint type, axis, origin, and limit.

Important points:

- Ignore traces in the background.
- Sometimes, cotracker might fail to capture traces of moving parts especially when the parts is moving forward and backward. Do your best to detect motion on your own.
- Traces moving in an arc indicates a revolute joint while linear traces indicate a prismatic joint.
- I will tip $200 for each correct analysis of the motion traces.
"""

JOINT_CRITIC_EXAMPLES = """
```json
{"gt_description": "The gt video shows the window pane opens by sliding horizontally along the y-axis in a linear motion.",
 "pred_description": "The pred video shows the window pane opens by sliding horizontally along the frame in a linear motion.",
 "candidate_function_description": "The `candidate_function` has `make_prismatic_joint` and axis is [0, -bbox[`width`], 0], which is horizontal (y-axis) and correct",
 "failure_reason": "success",
 "improvement_suggestion": "None",
 "realism_rating": 10
}
```
```json
{
"gt_description": "The gt video shows the window pane opens by sliding horizontally along the y-axis in a linear motion",
"pred_description": "The pred video shows the window opens by rotating up in an arc.",
"candidate_function_description": "The `candidate_function` has `make_revolute_joint`, which is incorrect.",
"failure_reason": "joint_type",
"improvement_suggestion": "Consider changing the joint type to prismatic to allow sliding motion",
"realism_rating": 0
}
```
```json
{
"gt_description": "The gt video shows the window pane opens by slides horizontally along the y-axis in a linear motion",
"pred_description": "The pred video looks static. We need to see the `candidate_function` to understand the issue.",
"candidate_function_description": "We have `make_prismatic_joint` which is correct but the axis `upper_point=[bbox[`length`], 0, 0]` is along x-axis (front/back) instead of y-axis (left/right).",
"failure_reason": "joint_axis",
"improvement_suggestion": "Consider changing `joint_axis` to slide along the y-axis",
"realism_rating": 1
}
```
```json
{"gt_description":   "The gt video shows the door opens by rotating forward along the vertical axis (z) while the **RIGHT** part fixed to the body",
 "pred_description":  "The pred video shows the door opens by rotating forward along the vertical axis (z) while the **LEFT** part fixed to the body",
 "candidate_function_description": "The `candidate_function` has `make_revolute_joint` and axis is [0, 0, 1], which is vertical (z-axis) and correct. The pivot point is set to Bottom-Front-**RIGHt** which is incorrect. Note that in the groundtruth, the left part of the door is fixed to the body.",
 "failure_reason": "joint_origin",
 "improvement_suggestion": "Try changing the pivot to the left side of the door (e.g. Front-**lEFT**-Bottom) to make the joint more like the groundtruth video.",
 "realism_rating": 2
}
```
```json
{"gt_description":   "The gt video shows the door opens outward. The door rotates outward along the vertical axis (z) while the left part fixed to the body",
 "pred_description":  "The pred video shows the door opens by rotating **inward** along the vertical axis (z) while the left part fixed to the body. The prediction doesn't look similar to the groundtruth as the door appears to be moving inward into the body instead of outward.",
 "candidate_function_description": "The `candidate_function` has `make_revolute_joint` and axis is [0, 0, 1], which is vertical (z-axis). The pivot point is set to Front-Left-Bottom which is correct, keeping the left part of the door fixed to the body. However, the door opens inward instead of outward so this is a joint limit issue.",
 "failure_reason": "joint_limit",
 "improvement_suggestion": "In our convention, left is negative so in order to open outward, the axis must be negative: i.e. [0, 0, -1]. The current axis is [0, 0, 1]. Try negating it",
 "realism_rating": 3
}
```

Important points:

- These examples are far from exhaustive. Use them as a guide to evaluate the realism of the joint.
- Use your own judgement to evaluate. Reason step-by-step.

"""


JOINT_CRITIC_EXAMPLES_SIMFOUNDRY = """
```json
{"gt_description": "The gt image shows a cabinet with three drawers",
 "frame_by_frame_analysis": "Frame 1: All three drawers are closed, flush with cabinet body. Frame 2: No movement visible. Frame 3: No movement visible. Frame 4: Still no movement. Frame 5: Drawers remain completely static.",
 "observed_motion_type": "no_visible_motion",
 "pred_description": "No motion is visible in the prediction video - all parts remain static",
 "candidate_function_description": "The `candidate_function` has `make_prismatic_joint` but the video shows no actual movement",
 "failure_reason": "no_prediction",
 "improvement_suggestion": "The simulation failed to produce motion - check if joints are properly configured",
 "realism_rating": 0
}
```
```json
{"gt_description": "The gt image shows a cabinet with a drawer that should slide out",
 "frame_by_frame_analysis": "Frame 1: Drawer is closed. Frame 2: Drawer has moved 2cm to the right in a straight line. Frame 3: Drawer continues right, now 5cm out. Frame 4: Drawer at 8cm, still moving linearly. Frame 5: Drawer fully extended 10cm in a perfectly straight horizontal line.",
 "observed_motion_type": "sliding_linear",
 "pred_description": "The drawer slides outward horizontally in a straight line to the right",
 "candidate_function_description": "The `candidate_function` has `make_prismatic_joint` with axis=[0, 1, 0] (y-axis/right), which matches the observed horizontal sliding motion",
 "failure_reason": "success",
 "improvement_suggestion": "None",
 "realism_rating": 10
}
```
```json
{"gt_description": "The gt image shows a door that should open by rotating",
 "frame_by_frame_analysis": "Frame 1: Door closed. Frame 2: Door has rotated 15° in an arc. Frame 3: Door at 30° rotation. Frame 4: Door at 45°. Frame 5: Door at 60° - clear rotating arc motion.",
 "observed_motion_type": "rotating_arc",
 "pred_description": "The door rotates in an arc around its left edge",
 "candidate_function_description": "The `candidate_function` has `make_prismatic_joint` which is WRONG - the video clearly shows rotation, not sliding",
 "failure_reason": "joint_type",
 "improvement_suggestion": "Change to make_revolute_joint to match the observed rotating motion",
 "realism_rating": 0
}
```
```json
{
"gt_description": "The gt image shows a window pane that opens by sliding horizontally along the y-axis in a linear motion",
"pred_description": "The pred video shows the window opens by rotating up in an arc.",
"candidate_function_description": "The `candidate_function` has `make_revolute_joint`, which is incorrect.",
"failure_reason": "joint_type",
"improvement_suggestion": "Consider changing the joint type to prismatic to allow sliding motion",
"realism_rating": 0
}
```
```json
{
"gt_description": "The gt image shows a window pane that opens by slides horizontally along the y-axis in a linear motion",
"pred_description": "The pred video looks static. We need to see the `candidate_function` to understand the issue.",
"candidate_function_description": "We have `make_prismatic_joint` which is correct but the axis `upper_point=[bbox[`length`], 0, 0]` is along x-axis (front/back) instead of y-axis (left/right).",
"failure_reason": "joint_axis",
"improvement_suggestion": "Consider changing `joint_axis` to slide along the y-axis",
"realism_rating": 1
}
```
```json
{"gt_description":   "The gt image shows a door that opens by rotating forward along the vertical axis (z) while the **RIGHT** part fixed to the body",
 "pred_description":  "The pred video shows the door opens by rotating forward along the vertical axis (z) while the **LEFT** part fixed to the body",
 "candidate_function_description": "The `candidate_function` has `make_revolute_joint` and axis is [0, 0, 1], which is vertical (z-axis) and correct. The pivot point is set to Bottom-Front-**RIGHt** which is incorrect. Note that in the groundtruth, the left part of the door is fixed to the body.",
 "failure_reason": "joint_origin",
 "improvement_suggestion": "Try changing the pivot to the left side of the door (e.g. Front-**lEFT**-Bottom) to make the joint more like the groundtruth video.",
 "realism_rating": 2
}
```
```json
{"gt_description":   "The gt iamge shows the door opens outward. The door rotates outward along the vertical axis (z) while the left part fixed to the body",
 "pred_description":  "The pred video shows the door opens by rotating **inward** along the vertical axis (z) while the left part fixed to the body. The prediction doesn't look similar to the groundtruth as the door appears to be moving inward into the body instead of outward.",
 "candidate_function_description": "The `candidate_function` has `make_revolute_joint` and axis is [0, 0, 1], which is vertical (z-axis). The pivot point is set to Front-Left-Bottom which is correct, keeping the left part of the door fixed to the body. However, the door opens inward instead of outward so this is a joint limit issue.",
 "failure_reason": "joint_limit",
 "improvement_suggestion": "In our convention, left is negative so in order to open outward, the axis must be negative: i.e. [0, 0, -1]. The current axis is [0, 0, 1]. Try negating it",
 "realism_rating": 3
}
```

Important points:

- These examples are far from exhaustive. Use them as a guide to evaluate the realism of the joint.
- Use your own judgement to evaluate. Reason step-by-step.

"""


class JointCritic(Agent):
    OUT_RESULT_PATH = "joint_critic.json"

    def _make_system_instruction(self):
        """
        ## General Instructions
        {...}
        {## CoTracker Motion Tracing}
        {If use_cotracker is True}

        {## Examples. Only for `basic` prompting}
        {...}
        """
        system_instruction = CRITIC_INSTRUCTION
        if self.cfg.joint_critic.use_cotracker:
            system_instruction += CRITIC_COTRACKER_TRACE

        if self.cfg.joint_critic.type == "basic":
            system_instruction += (
                "\n## Examples \n \n Here are some examples of the evaluation of the realism of various joints\n"
                + JOINT_CRITIC_EXAMPLES
                + "\n")
        return system_instruction

    def _make_prompt_parts(
        self,
        candidate_function_path: os.PathLike,
        prompt_path: os.PathLike,
        pred_video_path: os.PathLike,
        num_frames=5,
        video_encoding_strategy="individual",
    ):
        gt_video = get_frames_from_video(
            prompt_path,
            num_frames=num_frames,
            video_encoding_strategy=video_encoding_strategy,
            # width=self.cfg.simulator.camera_params.width,
            # height=self.cfg.simulator.camera_params.height,
        )
        pred_video = get_frames_from_video(
            pred_video_path,
            num_frames=num_frames,
            video_encoding_strategy=video_encoding_strategy,
            # width=self.cfg.simulator.camera_params.width,
            # height=self.cfg.simulator.camera_params.height,
        )
        candidate_function = file_to_string(candidate_function_path)
        candidate_function_text = (
            "The candidate function is:\n"
            + "```python\n"
            + candidate_function
            + "\n```"
        )
        prompt_parts = ["The groundtruth video is:\n"] + gt_video
        prompt_parts += ["The prediction video is:\n"] + pred_video
        prompt_parts += [candidate_function_text]
        return prompt_parts

    def parse_response(self, response, realign_score=True, **kwargs):
        # Extract the JSON string from the response text
        text = response.text.strip()
        
        # Find JSON between ```json and ``` markers
        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
            else:
                json_str = text[start:].strip()
        elif "```" in text:
            # Handle case where it's just ``` without json tag
            start = text.find("```") + len("```")
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
            else:
                json_str = text[start:].strip()
        else:
            # No code fence, try to find raw JSON
            json_str = text

        # Parse the JSON string into a dictionary
        parsed_response = json.loads(json_str, strict=False)

        if realign_score:
            scores = {
                "success": 10,
                "joint_type": 0,
                "joint_axis": 1,
                "joint_origin": 2,
                "joint_limit": 3,
                "no_prediction": 0,  # Missing video gets lowest score
            }
            parsed_response["realism_rating"] = scores.get(
                parsed_response["failure_reason"], 0
            )
            if int(parsed_response["realism_rating"]) > 5:
                parsed_response["failure_reason"] = "success"

        logging.info(f"Joint critic response: {parsed_response}")

        # Save the parsed response to a JSON file
        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))

        return parsed_response


class JointCriticSimfoundry(JointCritic):

    def _make_system_instruction(self):

        system_instruction = CRITIC_INSTRUCTION_SIMFOUNDRY
        if self.cfg.joint_critic.use_cotracker:
            system_instruction += CRITIC_COTRACKER_TRACE

       
        system_instruction += (
            "\n## Examples \n \n Here are some examples of the evaluation of the realism of various joints\n"
            + JOINT_CRITIC_EXAMPLES
            + "\n")
        return system_instruction

    def _make_prompt_parts(
        self,
        candidate_function_path: os.PathLike,
        prompt_path: os.PathLike,
        pred_video_path: os.PathLike,
        num_frames=8,
        video_encoding_strategy="individual",
    ):

        gt_image = [Image.open(prompt_path)]
        # Build prompt: GT image → Prediction video → Then candidate function
        # This order prevents the model from being biased by the function code
        prompt_parts = ["The groundtruth image is:\n"] + gt_image
        

        pred_video = get_frames_from_video(
            pred_video_path,
            num_frames=num_frames,
            video_encoding_strategy=video_encoding_strategy,
            # width=self.cfg.simulator.camera_params.width,
            # height=self.cfg.simulator.camera_params.height,
        )
        # pred_video = ["test image"]
        prompt_parts += [
                "\n ANALYZE THE PREDICTION VIDEO CAREFULLY:\n"
                "Look at each frame below and describe EXACTLY what you see. "
                "Focus on positions, movements, and trajectories.\n"
            ] + pred_video

        candidate_function = file_to_string(candidate_function_path)
        candidate_function_text = (
            "\n\n Now that you've analyzed the video, here is the candidate function:\n"
            + "```python\n"
            + candidate_function
            + "\n```\n"
            + "Does the code match what you ACTUALLY SAW in the video?"
        )
        prompt_parts += [candidate_function_text]
        
        return prompt_parts

class JointCriticSimfoundryVideo(JointCriticSimfoundry):

    def make_prompt_parts(self, *args, **kwargs):
        prompt_parts = self._make_prompt_parts(*args, **kwargs)
        save_prompt_parts_as_html_simfoundry(
            prompt_parts, join_path(self.cfg.out_dir, "prompt.html")
        )
        return prompt_parts

    def _make_system_instruction(self):

        system_instruction = CRITIC_INSTRUCTION_SIMFOUNDRY_VIDEO
        if self.cfg.joint_critic.use_cotracker:
            system_instruction += CRITIC_COTRACKER_TRACE

       
        system_instruction += (
            "\n## Examples \n \n Here are some examples of the evaluation of the realism of various joints\n"
            + JOINT_CRITIC_EXAMPLES
            + "\n")
        return system_instruction

    def _make_prompt_parts(
        self,
        candidate_function_path: os.PathLike,
        prompt_path: os.PathLike,
        pred_video_path: os.PathLike,
        num_frames=8,
        video_encoding_strategy="individual",
    ):

        gt_image = [Image.open(prompt_path)]
        with open(prompt_path, "rb") as f:
            gt_image_bytes = f.read()
        # Build prompt: GT image → Prediction video → Then candidate function
        # This order prevents the model from being biased by the function code
        # prompt_parts = ["The groundtruth image is:\n"] + gt_image


        
        with open(pred_video_path, "rb") as f:
            video_bytes = f.read()
        # pred_video = get_frames_from_video(
        #     pred_video_path,
        #     num_frames=num_frames,
        #     video_encoding_strategy=video_encoding_strategy,
        #     # width=self.cfg.simulator.camera_params.width,
        #     # height=self.cfg.simulator.camera_params.height,
        # )
        # pred_video = ["test image"]
        # prompt_parts += [
        #         "\n ANALYZE THE PREDICTION VIDEO CAREFULLY:\n"
        #         "Look at each frame below and describe EXACTLY what you see. "
        #         "Focus on positions, movements, and trajectories.\n"
        #     ] + pred_video

        candidate_function = file_to_string_python_prediction(candidate_function_path)
        # candidate_function_text = (
        #     "\n\n Now that you've analyzed the video, here is the candidate function:\n"
        #     + "```python\n"
        #     + candidate_function
        #     + "\n```\n"
        #     + "Does the code match what you ACTUALLY SAW in the video?"
        # )
        # prompt_parts += [candidate_function_text]

        parts = [
            types.Part(text="The groundtruth image is:\n"),
            types.Part.from_bytes(
                data=gt_image_bytes,
                mime_type="image/png",
            ),
            types.Part(text="The prediction video is:\n"),
            types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
            ),
            types.Part(text="The candidate function is:\n"),
            types.Part(text=candidate_function),
        ]
        
        return parts
   



class JointCriticMultiModalExamples(InContextExampleModel, JointCritic):
    def _make_system_instruction(self):
        return JointCritic._make_system_instruction(self)

    def get_example_paths(self):
        # paths under `examples_dir/{failure_reason}/{obj_id}/{joint_id}`
        pattern = join_path(self.cfg.project_root,
            self.cfg.joint_critic.examples_dir, "*", "*", "*")
        return [path for path in glob.glob(pattern) if os.path.isdir(path)]

    def _format_content(self, *args, **kwargs):
        joint_formatter = JointCritic(
            create_task_config(self.cfg, "temp"))
        expected_joint_critic_path = kwargs.pop(
            "expected_joint_critic_path", None)
        content = joint_formatter._make_prompt_parts(*args, **kwargs)
        if expected_joint_critic_path is None:
            return content
        joint_critic_response = load_json(expected_joint_critic_path)
        content.append(
            f"The correct response is:\n```json\n{json.dumps(joint_critic_response, indent=2)}\n```"
        )

        return content

    def _extract_example_kwargs(self, example_path):
        joint_id = os.path.basename(example_path)
        obj_id = os.path.basename(os.path.dirname(example_path))

        semantic_joint_id = get_semantic_joint_id(
            obj_id, joint_id,
            # input_dir=os.path.dirname(self.cfg.dataset_dir),
        )

        gt_video_name = (
            f"{'aug_' if self.cfg.joint_critic.use_cotracker else ''}video_{joint_id}_{self.cfg.cam_view}.mp4"
        )
        pred_video_name = f"{'aug_' if self.cfg.joint_critic.use_cotracker else ''}video_{semantic_joint_id}_{self.cfg.cam_view}.mp4"

        candidate_function_path = join_path(example_path, "joint_pred.py")
        expected_joint_critic_path = join_path(
            example_path, "joint_critic.json")

        gt_video_path = join_path(example_path, gt_video_name)
        pred_video_path = join_path(example_path, pred_video_name)

        return {
            "candidate_function_path": candidate_function_path,
            "gt_video_path": gt_video_path,
            "pred_video_path": pred_video_path,
            "expected_joint_critic_path": expected_joint_critic_path,
        }


def make_joint_critic(cfg):
    JointCriticCls = {
        "basic": JointCritic,
        "simfoundry": JointCriticSimfoundry,
        "simfoundry_video": JointCriticSimfoundryVideo,
        "incontext": JointCriticMultiModalExamples,
    }
    return JointCriticCls[cfg.joint_critic.type]
