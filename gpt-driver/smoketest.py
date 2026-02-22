from pathlib import Path
import openai

openai.api_key = Path(r"C:\Users\tnkln\Documents\OpenAI_API_key.txt").read_text().strip()

resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=5,
    temperature=0,
)
print(resp["choices"][0]["message"]["content"])
