import agentops

agentops.init(
    api_key="dummy",
    exporter_endpoint="http://localhost:8000/v1/traces",
)

import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello, world!"}],
)

print(response)
