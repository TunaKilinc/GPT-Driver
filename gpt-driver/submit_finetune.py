from pathlib import Path
from openai import OpenAI

# KEY FILE: put your key in a text file, one line only
KEY_PATH = Path(r"C:\Users\tnkln\Documents\OpenAI_API_key.txt")  # same folder as this script (recommended)

api_key = KEY_PATH.read_text(encoding="utf-8").strip()
if not api_key.startswith("sk-"):
    raise ValueError("API key file does not look like an OpenAI key. Ensure it starts with 'sk-' and contains only the key.")

client = OpenAI(api_key=api_key)

TRAIN_FILE = Path("data/train.json")  # because working dir is gpt-driver

uploaded = client.files.create(
    file=TRAIN_FILE.open("rb"),
    purpose="fine-tune"
)

job = client.fine_tuning.jobs.create(
    training_file=uploaded.id,
    model="gpt-3.5-turbo",
    hyperparameters={"n_epochs": 1},
)

print("Fine-tune job id:", job.id)
