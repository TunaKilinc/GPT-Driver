import openai
import pickle
import json
import ast
import numpy as np
import time
import argparse
import re
from pathlib import Path
from prompt_message import system_message, generate_user_message, generate_assistant_message
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

parser = argparse.ArgumentParser(description="GPT-Driver test.")
parser.add_argument("-i", "--id", type=str, help="GPT model id")
parser.add_argument("-o", "--output", type=str, help="output file name")
args = parser.parse_args()

saved_traj_name = "outputs/" + args.output + ".pkl"
saved_text_name = "outputs/" + args.output + "_text.pkl"
temp_text_name = "outputs/" + args.output + "_temp.jsonl"

KEY_PATH = Path(r"C:\Users\tnkln\Documents\OpenAI_API_key.txt")

api_key = KEY_PATH.read_text(encoding="utf-8").strip()
if not api_key.startswith("sk-"):
    raise ValueError("API key file does not look valid. Ensure the TXT file contains only the key and starts with 'sk-'.")

openai.api_key = api_key


data = pickle.load(open('data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('data/split.json', 'r'))

train_tokens = split["train"]
test_tokens = split["val"]

text_dict, traj_dict = {}, {}

invalid_tokens = []

untest_tokens = [
]

for token in test_tokens:
    if len(untest_tokens) > 0 and token not in untest_tokens: 
        continue

    print()
    print(token)

    time.sleep(1)    
    user_message = generate_user_message(data, token)
    assitant_message = generate_assistant_message(data, token)
    model_id = args.id
    completion = completion_with_backoff(
        model=model_id,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )
    # import pdb; pdb.set_trace()
    result = completion.choices[0].message["content"]
    print(f"GPT  Planner:\n {result}")
    print(f"Ground Truth:\n {assitant_message}")
    output_dict = {
        "token": token,
        "GPT": result,
        "GT": assitant_message, 
    }

    text_dict[token] = result

    # Try to extract trajectory list from anywhere in the text
    m = re.search(r"\[[\s\S]*?\]", result)  # first [...] block
    if not m:
        print(f"Invalid token (no traj found): {token}")
        invalid_tokens.append(token)
        continue

    traj_text = m.group(0)

    try:
        traj = ast.literal_eval(traj_text)  # should become a Python list/tuple
        # If model included a starting (0,0) and returned 7 points, drop the first one
        if isinstance(traj, (list, tuple)) and len(traj) == 7:
            if isinstance(traj[0], (list, tuple)) and len(traj[0]) == 2:
                if float(traj[0][0]) == 0.0 and float(traj[0][1]) == 0.0:
                    traj = traj[1:]
        traj = np.array(traj, dtype=float)
        # Optional strict check: must be 6 waypoints
        if traj.shape != (6, 2):
            raise ValueError(f"Unexpected traj shape: {traj.shape}")
    except Exception as e:
        print(f"Invalid token (parse error): {token} | {e}")
        invalid_tokens.append(token)
        continue

    traj_dict[token] = traj

    with open(temp_text_name, "a+") as file:
        file.write(json.dumps(output_dict) + '\n')

    # output_dicts = []
    # with open(temp_text_name, "r") as file:
    #     for line in file:
    #         output_dicts.append(json.loads(line))

    if len(untest_tokens) > 0:
        exist_dict = pickle.load(open(saved_traj_name, 'rb'))
        exist_dict.update(traj_dict)
        fd = open(saved_traj_name, "wb")
        pickle.dump(exist_dict, fd)

print("#### Invalid Tokens ####")
for token in invalid_tokens:
    print(token)

if len(untest_tokens) == 0:
    with open(saved_text_name, "wb") as f:
        pickle.dump(text_dict, f)
    with open(saved_traj_name, "wb") as f:
        pickle.dump(traj_dict, f)