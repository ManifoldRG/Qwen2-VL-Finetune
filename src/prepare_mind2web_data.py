import json
import os

TRAJECTORIES_PATH = "data/trajectory.json"
SCREENSHOTS_PATH = "data/screenshots"
FINETUNE_DATA_PATH = "data"
UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
## User Instruction
{instruction}
"""

with open(TRAJECTORIES_PATH, 'r') as f:
    trajectory_data = json.load(f)

print(f"Successfully loaded {len(trajectory_data)} items from {TRAJECTORIES_PATH}")

training_data = []
for i, example in enumerate(trajectory_data):
  screenshot = example.get("screenshot")
  step_instruction = example.get("step_instruction")
  op = example.get("op")
  coordinates = example.get("coordinates")
  type_action_value = example.get("type_action_value")
  if screenshot is None or step_instruction is None or op is None or coordinates is None: # Skip if any of these necessary fields is missing
    continue

  #Format prompt and prediction
  prompt = f"<image>\n${UITARS_USR_PROMPT_NOTHOUGHT.format(instruction = step_instruction)}"

  #Mind2Web has actions: Click, Type, Hover, Press Enter, Click (Fake) and Ignore. Map these actions to the UI Tars actions
  prediction = ""
  if op.lower() == "click" or op.lower == "hover":
    prediction = f"<image>\nAction: click(start_box='(${coordinates[0]}, ${coordinates[1]})')"
  elif op.lower() == "type":
    prediction = f"<image>\nAction: type(content='({type_action_value})')"
  elif op.lower() == "press enter":
    prediction = f"<image>\nAction: type(content='(\\n)')"   
  elif op.lower() == "ignore":
    prediction = f"<image>\nAction: wait()"   

  filename = os.path.join(SCREENSHOTS_PATH, os.path.basename(screenshot))

  training_data.append({
      "id": str(i),
      "image": filename,
      "conversations": [{
          "from": "human",
          "value": prompt
      },
      {
          "from": "gpt",
          "value": prediction,
      }]
  })

# Ensure the output directory exists
os.makedirs(FINETUNE_DATA_PATH, exist_ok=True)

# Define the output file path
output_file_path = os.path.join(FINETUNE_DATA_PATH, "training_data.json")

# Write the training_data to the JSON file
with open(output_file_path, 'w') as f:
    json.dump(training_data, f, indent=4)

print(f"Successfully wrote training data to {output_file_path}")