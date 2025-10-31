# Serving LLMs under Agent-lightning

Agent-lightning focuses on data, learning signals, and control flow — **not** on running model inference. This deep dive explains how to **serve** a model alongside Agent-lightning so runners can call it reliably, how the **LLM Proxy** fits into the loop, and why **token IDs** matter if you care about correctness in training and evaluation.

## General background on LLM serving

[](){ #general-llm-serving-background }

Serving a model is essential if you want to train it, especially when you use the model’s own generations as training data. We’ll briefly review the general background to ensure all readers are aligned.

Modern LLM servers solve a difficult scheduling problem: keeping GPUs fully utilized while handling prompts of different lengths, streaming tokens as they arrive, and fitting large KV caches into limited memory. Techniques like [**continuous batching**](https://www.anyscale.com/blog/continuous-batching-llm-inference) and [**paged attention**](https://arxiv.org/abs/2309.06180) address these challenges. Continuous batching interleaves decoding across requests to reuse weights efficiently; with careful memory planning, it achieves major throughput gains without increasing latency. PagedAttention reduces KV-cache fragmentation so batching remains effective as sequences grow. See [vLLM’s PagedAttention paper](https://arxiv.org/abs/2309.06180) and [industry analyses](https://www.baseten.co/blog/continuous-vs-dynamic-batching-for-ai-inference/) for details. Balancing inference correctness and efficiency is difficult — a [recent blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) from Thinking Machines Labs highlights how inference nondeterminism ultimately affects training.

Beyond scheduling, servers expose an HTTP API, often **OpenAI-compatible** (`/v1/chat/completions` and `/v1/responses`), which is itself a complex stack. In addition to text prompts and chat messages, the API defines many parameters and response fields such as [tool calls](https://platform.openai.com/docs/guides/function-calling), [structured output](https://platform.openai.com/docs/guides/structured-outputs), and [multimodal support](https://platform.openai.com/docs/guides/images-vision). Much effort has been put into implementing all these parameters for many frameworks. Popular engines like **vLLM** and [**SGLang**](https://github.com/sgl-project/sglang) ship with OpenAI-compatible frontends so you can reuse existing client code. [Ollama](https://ollama.com/blog/openai-compatibility) and [llama.cpp](https://llama-cpp-python.readthedocs.io/en/latest/server/) provide similar capabilities. However, because models differ internally, each framework interprets and implements the API slightly differently. Even with identical requests, the tokens passed to the model can vary substantially across frameworks.

## What Agent-lightning expects from a served LLM

Most of the issues above either have workarounds or remain open research problems. Keep them in mind, but the key question is: what does Agent-lightning expect from a served LLM? The answer includes at least two things:

* An OpenAI-compatible **Chat Completions** or **Responses** endpoint the agent can call during rollouts.
* Optional training and debugging signals: **logprobs**, **usage**, and ideally **token IDs**. (OpenAI’s public API exposes usage and logprobs, but **not** token IDs — more on [why IDs matter][token-ids-matter] later.)

## Launching a serving framework

For many algorithms, you’ll start an engine (e.g., **vLLM** or **SGLang**) before rollouts, then shut it down afterward to free GPU memory. Most frameworks provide a one-line “serve” command to launch the OpenAI-compatible server. You can use those to bring up `/v1/chat/completions` with your checkpoint, ensuring streaming and any required tool-calling features are enabled. A working example is shown in [Unsloth SFT](../how-to/unsloth-sft.md).

Weight updates — which occur after each training step — are trickier. Some frameworks like [vLLM](https://vllm.ai/) support hot-updating model weights, but it’s usually simpler and more reliable to restart the engine to load new weights. For medium-sized tasks (hundreds of rollouts taking 10+ minutes), the restart overhead (under 30 seconds) is typically negligible.

If you’re using Agent-lightning’s [**VERL**][agentlightning.algorithm.verl.VERL] integration, the algorithm can **manage the server automatically**. The [VERL framework](https://github.com/volcengine/verl) intelligently allocates compute resources and wraps vLLM/SGLang behind an `AsyncLLMServer` abstraction. You can directly use this as the LLM endpoint for agents. Since VERL can spawn multiple vLLM replicas, using [`LLMProxy`][agentlightning.LLMProxy] to manage them adds an additional safety layer.

A full sequence diagram of how [VERL][agentlightning.algorithm.verl.VERL] interacts with the LLM server and proxy is available [here][birds-eye-view-verl-example].

## LLM Proxy

The **LLM Proxy** is a utility class in Agent-lightning, built on [LiteLLM](https://docs.litellm.ai/), that sits between runners and your backend engine(s) or server(s). In Agent-lightning it acts as a single URL registered as a [`Resource`][agentlightning.Resource] in the store, offering three key benefits:

1. **Unified endpoint & hot-swaps.** You can redirect traffic between OpenAI, Anthropic, local vLLM/SGLang, or canary checkpoints without modifying agent code — simply repoint the proxy.
2. **First-class tracing.** The proxy emits **OpenTelemetry** spans for every call and sends them to the [`LightningStore`][agentlightning.LightningStore]. It includes rollout and attempt identifiers in request headers so spans are correctly attributed. Sequence numbers are allocated monotonically via the store to [prevent clock-skew issues][distributed-tracing] and allow reliable reconstruction of execution trees.
3. **Token IDs.** The proxy can return prompt and response token IDs along with the model output. More details are available in the [next section][token-ids-matter].

Operationally, running the proxy alongside the algorithm works best: the algorithm registers the backend (e.g., the vLLM URL) via [`LLMProxy.update_model_list`][agentlightning.LLMProxy.update_model_list], publishes the proxy URL as a resource via [`LightningStore.add_resources`][agentlightning.LightningStore.add_resources], and runners simply use that URL during rollouts. This mirrors many production client–server setups.

## Token IDs and why they matter

[](){ #token-ids-matter }

This section explains how Agent-lightning handles and uses token IDs — a subtle but important detail for training stability and accuracy.

Most agents interact with LLMs via **Chat Completion APIs**, exchanging chat messages. There are two main approaches to collecting training data from such agents.

!!! note

    Tokenization here refers to the process of converting **Chat Messages** into **Token IDs**. Detokenization is the reverse process of converting **Token IDs** back to **Chat Messages**. Normally, the tokenizer is published along with the pretrained model, which includes a vocabulary, special tokens, and a chat template to dealing with chat messages.

**1. Retokenizing chat messages.**
In this approach, you store chat messages as text and let training algorithms **retokenize** them later, as done in many SFT workflows (e.g., [HuggingFace SFT](https://huggingface.co/docs/trl/sft_trainer)).
In practice, we’ve found this method unstable and less accurate. The chart below compares training results. The retokenization approach is run twice. All settings are the same except for the retokenization approach.

<div style="height:400px">
<canvas data-chart='{
  "type": "line",
  "data": {
    "labels": [0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0],
    "datasets": [
      {
        "label": "With Token IDs from Retokenization",
        "data": [0.49, 0.512, 0.54, 0.532, 0.54, 0.466, 0.328, 0.358, 0.348, 0.35, 0.346, 0.372, 0.346, 0.33, 0.346, 0.332],
        "spanGaps": true
      },
      {
        "label": "Retokenization (Second Run)",
        "data": [0.494, 0.526, 0.536, 0.554, 0.544, 0.556, 0.568, 0.552, 0.45, 0.466, 0.474, 0.47, 0.464, 0.476, 0.488, 0.432],
        "spanGaps": true
      },
      {
        "label": "With Token IDs from Engine",
        "data": [0.494, 0.522, 0.514, 0.538, 0.53, 0.564, 0.564, 0.586, 0.594, 0.604, 0.618, 0.584, 0.606, 0.558, 0.612, 0.588],
        "spanGaps": true
      }
    ]
  },
  "options": {
    "interaction": {
      "mode": "nearest",
      "intersect": false
    },
    "plugins": {
      "legend": {
        "display": true,
        "position": "top"
      },
      "title": {
        "display": true,
        "text": "Agent Training Results Comparison"
      }
    },
    "scales": {
      "x": {
        "title": {
          "display": true,
          "text": "Step"
        }
      },
      "y": {
        "title": {
          "display": true,
          "text": "Reward"
        }
      }
    }
  }
}'></canvas>
</div>

This instability has three causes. Firstly, chat template used in different frameworks could be slightly different. For example, one single LLaMA model can work with multiple chat templates (multiple in [vLLM](https://github.com/vllm-project/vllm/tree/1d165d6d859d3c50720f0c07209db2363c4fd33b/examples) and one in [HuggingFace](https://huggingface.co/meta-llama)). It's possible that the chat template used in detokenization is different from the one used in tokenization (this is actually an implementation bug).

Secondly, a word might be generated as two tokens (e.g., `H + AVING`) but later retokenized as `HAV + ING`. The text looks identical, but the token IDs differ from what the model originally produced.

Thirdly, a generated tool call text like `<tool_call>{ "name": ... }</tool_call>` is parsed by tool call parser into an object that is required by chat completion API. Later, the object is rendered back to `<tool_call>{ "name": ... }</tool_call>` and retokenized again, tool call parsing and re-rendering might cause changes in whitespace and formatting. In some situations, JSON errors may even be auto-corrected by the tool call parser — masking the model’s true generation errors and preventing them from being trained away.

----

**2. Saving token IDs directly.**
The alternative is to save the token IDs generated by the model, as done in RL setups like [Tinker](https://thinkingmachines.ai/tinker/). This requires a training pipeline that treats tokens as first-class entities, meaning agents must communicate with the inference engine at the token level.

However, most agents — especially those built with frameworks like LangChain — rely on OpenAI-compatible APIs and can’t tokenize or detokenize themselves. As mentioned [earlier][general-llm-serving-background], implementing this layer manually is complex and error-prone. Some frameworks implement custom solutions (e.g., [VERL Agent Loop](https://github.com/volcengine/verl/blob/4da0d3d3188072772cb2ec817b3d6cf4a463821f/recipe/langgraph_agent/chat_model.py), [Tinker Renderer](https://github.com/thinking-machines-lab/tinker-cookbook/blob/34a6588d7055040c259985d98e71c0140b389ba7/tinker_cookbook/renderers.py)), while others leave it to users (e.g., [SkyRL Search-R1](https://novasky-ai.notion.site/skyrl-searchr1)).

----

A better solution is to use an **OpenAI-compatible API that returns token IDs directly.** This lets agents continue using familiar APIs while capturing token IDs via [tracing](../tutorials/traces.md) for training. The limitation, of course, is that the serving framework must actually support this capability.

When Agent-lightning was first released, we implemented an [instrumented vLLM server](https://github.com/microsoft/agent-lightning/blob/v0.1/agentlightning/instrumentation/vllm.py) that monkey-patched vLLM’s OpenAI server to return token IDs. Since then, the Agent-lightning and vLLM teams have collaborated to add this feature directly to [vLLM core](https://github.com/vllm-project/vllm/pull/22587). Starting with **vLLM v0.10.2**, the OpenAI-compatible API includes a [`return_token_ids` parameter](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html#api-reference), allowing token IDs to be requested alongside chat messages. SGLang has tracked [similar feature requests](https://github.com/sgl-project/sglang/issues/2634), though its OpenAI-compatible layer doesn’t yet support them.

In short, when using vLLM v0.10.2 or newer, [`LLMProxy`][agentlightning.LLMProxy] automatically adds `return_token_ids` to each request so the engine includes token IDs in its response. For older vLLM versions, you still need the instrumented version (via `agl vllm` CLI command).

Finally, if you only save token IDs in spans, it will have its own limitations — if you train one model using spans from another model with a different tokenizer, incompatibilities can arise. In practice, though, spans in Agent-lightning always store both chat messages and token IDs (actually the full request and response objects), allowing you to fall back to retokenization when necessary.
