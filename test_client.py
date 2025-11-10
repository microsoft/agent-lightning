from anthropic import Anthropic

client = Anthropic(api_key="dummy", base_url="http://localhost:8081/rollout/123/attempt/456/")

response = client.messages.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Hello, world!"}],
    max_tokens=10,
)

print(response.content[0].text)
