import openai
import pickle
import json
import ast
import re
import numpy as np
import time
import argparse
from pathlib import Path

from prompt_message import system_message, generate_user_message, generate_assistant_message
from tenacity import retry, stop_after_attempt, wait_random_exponential  # exponential backoff


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def extract_trajectory(result_text: str):
    """
    Extract the first [...] block from the model output and parse it as a Python literal.
    Returns np.ndarray of shape (6, 2) if successful; otherwise returns None and an error string.
    """
    # Find the first bracketed list in the output (trajectory usually appears as [ ... ])
    m = re.search(r"\[[\s\S]*?\]", result_text)
    if not m:
        return None, "no [ ... ] block found"

    traj_text = m.group(0)

    try:
        traj = ast.literal_eval(traj_text)
    except Exception as e:
        return None, f"ast.literal_eval failed: {e}"

    if not isinstance(traj, (list, tuple)):
        return None, "parsed trajectory is not a list/tuple"

    # If the model included a starting (0,0) and returned 7 points, drop the first one
    if len(traj) == 7:
        try:
            first = traj[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                if float(first[0]) == 0.0 and float(first[1]) == 0.0:
                    traj = traj[1:]
        except Exception:
            # If this check fails, just keep as-is and validate later
            pass

    # Convert to numpy
    try:
        traj_np = np.array(traj, dtype=float)
    except Exception as e:
        return None, f"np.array conversion failed: {e}"

    # Strict shape check expected by the benchmark
    if traj_np.shape != (6, 2):
        return None, f"unexpected trajectory shape: {traj_np.shape} (expected (6,2))"

    return traj_np, None


def main():
    parser = argparse.ArgumentParser(description="GPT-Driver test (fixed parsing).")
    parser.add_argument("-i", "--id", type=str, required=True, help="GPT model id (e.g., gpt-4o-mini)")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file name (no extension)")
    args = parser.parse_args()

    # Ensure outputs folder exists
    Path("outputs").mkdir(parents=True, exist_ok=True)

    saved_traj_name = "outputs/" + args.output + ".pkl"
    saved_text_name = "outputs/" + args.output + "_text.pkl"
    temp_text_name = "outputs/" + args.output + "_temp.jsonl"

    # Read API key from txt
    KEY_PATH = Path(r"C:\Users\tnkln\Documents\OpenAI_API_key.txt")
    api_key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not api_key.startswith("sk-"):
        raise ValueError(
            "API key file does not look valid. Ensure the TXT file contains only the key and starts with 'sk-'."
        )
    openai.api_key = api_key

    # Load data
    data = pickle.load(open("data/cached_nuscenes_info.pkl", "rb"))
    split = json.load(open("data/split.json", "r"))

    test_tokens = split["val"]

    text_dict, traj_dict = {}, {}
    invalid_tokens = []

    # Leave empty to test all tokens; or list tokens to run only a subset
    untest_tokens = []

    for token in test_tokens:
        if len(untest_tokens) > 0 and token not in untest_tokens:
            continue

        print("\n" + token)

        # Small delay to be gentle on rate limits
        time.sleep(0.5)

        user_message = generate_user_message(data, token)
        assitant_message = generate_assistant_message(data, token)

        model_id = args.id

        # Extra instruction to help formatting consistency
        format_guard = (
            "IMPORTANT OUTPUT FORMAT:\n"
            "Return ONLY the final trajectory as a Python list of 6 (x,y) tuples, like:\n"
            "[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]\n"
            "Do NOT add any text before or after the list."
        )

        completion = completion_with_backoff(
            model=model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "user", "content": format_guard},
            ],
        )

        result = completion.choices[0].message["content"]
        print(f"GPT  Planner:\n{result}")
        print(f"Ground Truth:\n{assitant_message}")

        output_dict = {"token": token, "GPT": result, "GT": assitant_message}
        text_dict[token] = result

        traj_np, err = extract_trajectory(result)
        if traj_np is None:
            print(f"Invalid token: {token} | {err}")
            invalid_tokens.append(token)
            continue

        traj_dict[token] = traj_np

        with open(temp_text_name, "a+", encoding="utf-8") as file:
            file.write(json.dumps(output_dict) + "\n")

    print("\n#### Invalid Tokens ####")
    for token in invalid_tokens:
        print(token)

    # Save results
    with open(saved_text_name, "wb") as f:
        pickle.dump(text_dict, f)
    with open(saved_traj_name, "wb") as f:
        pickle.dump(traj_dict, f)

    print(f"\nSaved text -> {saved_text_name}")
    print(f"Saved traj -> {saved_traj_name}")


if __name__ == "__main__":
    main()
