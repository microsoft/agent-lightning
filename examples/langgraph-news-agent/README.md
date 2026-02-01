# LangGraph News Agent with APO Training

This example demonstrates how to train a LangGraph-based news agent using Agent Lightning's Automatic Prompt Optimization (APO).

## Features

- **LangGraph integration** with Google Gemini 2.5 Flash
- **Multi-tool agent** (NewsAPI search + calculator)
- **Bulletproof placeholder handling** to prevent KeyErrors from APO
- **Persistent training results** (JSON, TXT, PKL formats)
- **Windows compatibility** fixes

## Included Files

| File | Purpose |
|------|---------|
| `agent_train.py` | Main training script with LangGraph agent and APO setup |
| `README.md` | This file - setup and usage instructions |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for API keys |

## Prerequisites

- Python 3.10+
- API keys for:
  - Google Gemini (https://aistudio.google.com/app/apikey)
  - OpenAI (https://platform.openai.com/api-keys)
  - NewsAPI (https://newsapi.org/account)

## Installation

1. **Clone and navigate:**
```bash
   cd examples/langgraph-news-agent
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
   cp .env.example .env
   # Edit .env and add your API keys
```

## Quick Start
```bash
python agent_train.py
```

Expected output:
```
Starting Agent Lightning training...
Training dataset size: 10
Validation dataset size: 2
Will run for 3 rounds...

[Round 01 | Prompt v1] Training on 10 samples...
[Round 01 | Prompt v1] Val score: 0.725
...
✅ Training complete!
Best validation score: 0.925
```

## Training Configuration

The script trains an agent that:
- Answers news queries using NewsAPI search
- Performs calculations using a calculator tool
- Optimizes via APO over 3 rounds with beam_width=2

Training data: 10 diverse tasks (news + math)
Validation data: 2 held-out tasks

## Output Files

After training:
- `optimized_prompt.txt` - The improved system prompt
- `training_results.json` - Full training metrics
- `trainer_state.pkl` - Complete trainer state

## Key Implementation Details

### Bulletproof Placeholder Handling

APO's GPT-4 editor may create prompts with unexpected placeholders. This example uses regex-based auto-detection:
```python
placeholders = re.findall(r'\{([^}]+)\}', prompt_template.template)
safe_vars = {"query": task["query"]}
for placeholder in placeholders:
    if placeholder not in safe_vars:
        safe_vars[placeholder] = ""  # Prevents KeyError
```

### Iteration Limiting

Set `beam_rounds` to prevent infinite training loops:
```python
algo = agl.APO(
    openai_client,
    beam_rounds=3  # Stops after 3 rounds
)
```

## Customization

**Use your own tools:**
```python
@tool
def my_custom_tool(query: str):
    # Your tool logic
    return result

tools = [my_custom_tool]
```

**Change the reward function:**
```python
def grade_response(final_response: str, task: NewsTask) -> float:
    # Your custom scoring logic
    return score  # 0.0 to 1.0
```

**Add more training data:**
```python
train_dataset = [
    NewsTask(query="...", expected_keywords=[...]),
    # Add 20-50 examples for best results
]
```

## Troubleshooting

**KeyError with placeholders:**
- The bulletproof placeholder handler should prevent this
- If it still occurs, check your prompt template syntax

**Infinite training loop:**
- Ensure `beam_rounds` is set in APO config
- Default behavior without this parameter is to run indefinitely

**Low validation scores:**
- Add more diverse training examples (aim for 20-50)
- Improve reward function to better capture quality
- Increase `beam_rounds` to 5

## References

- [Agent Lightning Documentation](https://microsoft.github.io/agent-lightning)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## License

This example is part of the Agent Lightning project and follows the same [MIT License](../../LICENSE).