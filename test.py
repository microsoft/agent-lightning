     
model="groq-meta-llama/llama-4-maverick-17b-128e-instruct"

parts = model.split("-", 1)
provider = parts[0]
actual_model = parts[1] if len(parts) > 1 else model

print(actual_model)