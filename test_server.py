import agentlightning as agl

llm_proxy = agl.LLMProxy(
    port=8081,
    model_list=[
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "litellm_params": {
                "model": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct",
                "api_base": "http://localhost:9090/v1",
            },
        }
    ],
    store=agl.InMemoryLightningStore(),
)

llm_proxy.start()
import time

time.sleep(1000000)
