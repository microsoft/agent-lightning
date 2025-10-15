# Write the First Algorithm with Agent-lightning

Running `python apo_custom_algorithm.py algo` will produce the following output:

```text
[Algo] Updating prompt template to: 'You are a helpful assistant. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-1d18988581cd' is now available for clients.
[Algo] Received Result: rollout_id='ro-1d18988581cd' input='Explain why the sky appears blue using principles of light scattering in 100 words.' start_time=1760451829.1076183
end_time=1760451835.3247516 mode='train' resources_id='rs-21c3dfb83535' status='succeeded' config=RolloutConfig(timeout_seconds=None, unresponsive_seconds=None, max_attempts=1,
retry_condition=[]) metadata={}
[LLM] Span 64879cb68069e41e (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.streaming': False,
'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a helpful assistant. Explain why the sky appears blue using principles of light scattering in 100 words.',
'gen_ai.response.id': 'chatcmpl-CQaEKc8rj9iDbX4VunV29AOWfFK20', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34',
'gen_ai.usage.total_tokens': 130, 'gen_ai.usage.prompt_tokens': 29, 'gen_ai.usage.completion_tokens': 101, 'gen_ai.completion.0.finish_reason': 'stop',
'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false, "severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text":
{"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity":
"safe"}}', 'gen_ai.completion.0.role': 'assistant', 'gen_ai.completion.0.content': "The sky appears blue because of a process called Rayleigh scattering. Sunlight contains all colors, but blue
light waves are shorter and scatter more easily when they hit molecules in Earth's atmosphere. This scattered blue light spreads in all directions, making the sky look blue to our eyes. During
sunrise or sunset, sunlight passes through more atmosphere, scattering away the blue and green wavelengths, leaving the reds and oranges visible. Thus, the scattering of shorter blue wavelengths
explains why the sky generally looks blue during the day.", 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": false, "severity":
"safe"}, "jailbreak": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered":
false, "severity": "safe"}}}]'}
[LLM] Span 715da3018f54f70b (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.temperature': 0.0,
'gen_ai.request.streaming': False, 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': "Evaluate how well the output fulfills the task.\nTask: Explain why the sky appears blue using
principles of light scattering in 100 words.\nOutput: The sky appears blue because of a process called Rayleigh scattering. Sunlight contains all colors, but blue light waves are shorter and
scatter more easily when they hit molecules in Earth's atmosphere. This scattered blue light spreads in all directions, making the sky look blue to our eyes. During sunrise or sunset, sunlight
passes through more atmosphere, scattering away the blue and green wavelengths, leaving the reds and oranges visible. Thus, the scattering of shorter blue wavelengths explains why the sky
generally looks blue during the day.\nYou must be very critical and strict in your evaluation.\nReturn only a number between 0 and 1. No text, punctuation, or explanation.", 'gen_ai.response.id':
'chatcmpl-CQaEM2mpRfBp0dySIsiyPm3mHFS7G', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 170,
'gen_ai.usage.prompt_tokens': 166, 'gen_ai.usage.completion_tokens': 4, 'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false,
"severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false,
"severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant',
'gen_ai.completion.0.content': '0.95', 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak":
{"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity":
"safe"}}}]'}
[Algo] Final reward: 0.95

[Algo] Updating prompt template to: 'You are a knowledgeable AI. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-5ca1724a8d28' is now available for clients.
[Algo] Received Result: rollout_id='ro-5ca1724a8d28' input='Explain why the sky appears blue using principles of light scattering in 100 words.' start_time=1760451836.2048416
end_time=1760451846.2557054 mode='train' resources_id='rs-fffdedcfcb0b' status='succeeded' config=RolloutConfig(timeout_seconds=None, unresponsive_seconds=None, max_attempts=1,
retry_condition=[]) metadata={}
[LLM] Span bdabfd3deb89d5eb (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.streaming': False,
'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a knowledgeable AI. Explain why the sky appears blue using principles of light scattering in 100 words.', 'gen_ai.response.id':
'chatcmpl-CQaEUkhYfUO7p03tEoXbI5lKPO1Cj', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 145,
'gen_ai.usage.prompt_tokens': 29, 'gen_ai.usage.completion_tokens': 116, 'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false,
"severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false,
"severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant',
'gen_ai.completion.0.content': "The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it interacts with molecules and tiny particles.
Shorter wavelengths of light, like blue and violet, scatter more efficiently than longer wavelengths such as red or yellow. Although violet light scatters even more, our eyes are less sensitive
to it, and some of it is absorbed by the upper atmosphere. As a result, the scattered blue light dominates, giving the sky its characteristic color. This scattering distributes blue light evenly
across the sky, making it appear blue to our eyes during the day.", 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": false, "severity":
"safe"}, "jailbreak": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered":
false, "severity": "safe"}}}]'}
[LLM] Span e78c522b3dba75c3 (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.temperature': 0.0,
'gen_ai.request.streaming': False, 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': "Evaluate how well the output fulfills the task.\nTask: Explain why the sky appears blue using
principles of light scattering in 100 words.\nOutput: The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it interacts with molecules
and tiny particles. Shorter wavelengths of light, like blue and violet, scatter more efficiently than longer wavelengths such as red or yellow. Although violet light scatters even more, our eyes
are less sensitive to it, and some of it is absorbed by the upper atmosphere. As a result, the scattered blue light dominates, giving the sky its characteristic color. This scattering distributes
blue light evenly across the sky, making it appear blue to our eyes during the day.\nYou must be very critical and strict in your evaluation.\nReturn only a number between 0 and 1. No text,
punctuation, or explanation.", 'gen_ai.response.id': 'chatcmpl-CQaEXVaf8k7Cb21ibNKTZ7Lpd8VKA', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint':
'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 185, 'gen_ai.usage.prompt_tokens': 181, 'gen_ai.usage.completion_tokens': 4, 'gen_ai.completion.0.finish_reason': 'stop',
'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false, "severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text":
{"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity":
"safe"}}', 'gen_ai.completion.0.role': 'assistant', 'gen_ai.completion.0.content': '0.95', 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate":
{"filtered": false, "severity": "safe"}, "jailbreak": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity":
"safe"}, "violence": {"filtered": false, "severity": "safe"}}}]'}
[Algo] Final reward: 0.95

[Algo] Updating prompt template to: 'You are a friendly chatbot. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-6be16b73678a' is now available for clients.
[Algo] Received Result: rollout_id='ro-6be16b73678a' input='Explain why the sky appears blue using principles of light scattering in 100 words.' start_time=1760451846.3379323
end_time=1760451856.7222157 mode='train' resources_id='rs-1b5fac8922e0' status='succeeded' config=RolloutConfig(timeout_seconds=None, unresponsive_seconds=None, max_attempts=1,
retry_condition=[]) metadata={}
[LLM] Span 60e14792bebf6054 (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.streaming': False,
'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a friendly chatbot. Explain why the sky appears blue using principles of light scattering in 100 words.', 'gen_ai.response.id':
'chatcmpl-CQaEg6lCLn8EV87xANMfvck7eKE7h', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 132,
'gen_ai.usage.prompt_tokens': 29, 'gen_ai.usage.completion_tokens': 103, 'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false,
"severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false,
"severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant',
'gen_ai.completion.0.content': "The sky appears blue because of a process called Rayleigh scattering. When sunlight enters Earth's atmosphere, it is made up of different colors, each with
different wavelengths. Blue light has a shorter wavelength than other colors, so it scatters more easily when it hits tiny molecules in the air. This scattered blue light spreads across the sky,
making the sky look blue to our eyes. During sunrise and sunset, the sunlight passes through more atmosphere, scattering away the blue and giving the sky its beautiful red and orange hues.",
'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak": {"filtered": false, "detected": false},
"self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}}]'}
[LLM] Span 85af9688650c4a17 (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.temperature': 0.0,
'gen_ai.request.streaming': False, 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': "Evaluate how well the output fulfills the task.\nTask: Explain why the sky appears blue using
principles of light scattering in 100 words.\nOutput: The sky appears blue because of a process called Rayleigh scattering. When sunlight enters Earth's atmosphere, it is made up of different
colors, each with different wavelengths. Blue light has a shorter wavelength than other colors, so it scatters more easily when it hits tiny molecules in the air. This scattered blue light
spreads across the sky, making the sky look blue to our eyes. During sunrise and sunset, the sunlight passes through more atmosphere, scattering away the blue and giving the sky its beautiful red
and orange hues.\nYou must be very critical and strict in your evaluation.\nReturn only a number between 0 and 1. No text, punctuation, or explanation.", 'gen_ai.response.id':
'chatcmpl-CQaEi1KDyAZ2fvKU0es3JuMJHex6z', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 172,
'gen_ai.usage.prompt_tokens': 168, 'gen_ai.usage.completion_tokens': 4, 'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false,
"severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false,
"severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant',
'gen_ai.completion.0.content': '0.95', 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak":
{"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity":
"safe"}}}]'}
[Algo] Final reward: 0.95

[Algo] All prompts and their rewards: [('You are a helpful assistant. {any_question}', 0.95), ('You are a knowledgeable AI. {any_question}', 0.95), ('You are a friendly chatbot. {any_question}',
0.95)]
[Algo] Best prompt found: 'You are a helpful assistant. {any_question}' with reward 0.95
Unclosed client session
client_session: <aiohttp.client.ClientSession object at 0x7853730e3150>
Unclosed connector
connections: ['deque([(<aiohttp.client_proto.ResponseHandler object at 0x785372b35b00>, 51958.07625564)])']
connector: <aiohttp.connector.TCPConnector object at 0x785373c62290>
❯ python apo_custom_algorithm.py algo
/home/kiki/Projects/agl-second-bench/agentlightning/reward.py:7: UserWarning: agentlightning.reward is deprecated. Please use agentlightning.emitter instead.
  warnings.warn("agentlightning.reward is deprecated. Please use agentlightning.emitter instead.")

[Algo] Updating prompt template to: 'You are a helpful assistant. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-de8496b948d0' is now available for clients.
[Algo] Received Result: rollout_id='ro-de8496b948d0' input='Explain why the sky appears blue using principles of light scattering in 100 words.' start_time=1760451934.6312773
end_time=1760451941.56359 mode='train' resources_id='rs-a2f1df8ba79f' status='succeeded' config=RolloutConfig(timeout_seconds=None, unresponsive_seconds=None, max_attempts=1, retry_condition=[])
metadata={}
[LLM] Span 72db848ac5a39723 (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.streaming': False,
'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a helpful assistant. Explain why the sky appears blue using principles of light scattering in 100 words.',
'gen_ai.response.id': 'chatcmpl-CQaG2xl23mgnbqKwdqKrTawdey7ua', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34',
'gen_ai.usage.total_tokens': 140, 'gen_ai.usage.prompt_tokens': 29, 'gen_ai.usage.completion_tokens': 111, 'gen_ai.completion.0.finish_reason': 'stop',
'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false, "severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text":
{"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity":
"safe"}}', 'gen_ai.completion.0.role': 'assistant', 'gen_ai.completion.0.content': 'The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight reaches Earth’s
atmosphere, it is composed of various colors with different wavelengths. Shorter wavelengths, like blue and violet, are scattered more effectively by tiny molecules in the air. Blue light is
scattered in all directions, making the sky appear predominantly blue to our eyes. Violet light is scattered even more but is less noticeable because our eyes are less sensitive to violet, and
some ultraviolet is absorbed by the atmosphere. This scattering effect explains the blue appearance of the sky during the day.', 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0,
"content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual":
{"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}}]'}
[LLM] Span b670307c550ca6c0 (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.temperature': 0.0,
'gen_ai.request.streaming': False, 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'Evaluate how well the output fulfills the task.\nTask: Explain why the sky appears blue using
principles of light scattering in 100 words.\nOutput: The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight reaches Earth’s atmosphere, it is composed of various
colors with different wavelengths. Shorter wavelengths, like blue and violet, are scattered more effectively by tiny molecules in the air. Blue light is scattered in all directions, making the
sky appear predominantly blue to our eyes. Violet light is scattered even more but is less noticeable because our eyes are less sensitive to violet, and some ultraviolet is absorbed by the
atmosphere. This scattering effect explains the blue appearance of the sky during the day.\nYou must be very critical and strict in your evaluation.\nReturn only a number between 0 and 1. No
text, punctuation, or explanation.', 'gen_ai.response.id': 'chatcmpl-CQaG4nM3cEqT8agTmuMCPF8NPxGDS', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint':
'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 180, 'gen_ai.usage.prompt_tokens': 176, 'gen_ai.usage.completion_tokens': 4, 'gen_ai.completion.0.finish_reason': 'stop',
'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false, "severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text":
{"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity":
"safe"}}', 'gen_ai.completion.0.role': 'assistant', 'gen_ai.completion.0.content': '0.95', 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate":
{"filtered": false, "severity": "safe"}, "jailbreak": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity":
"safe"}, "violence": {"filtered": false, "severity": "safe"}}}]'}
[Algo] Final reward: 0.95

[Algo] Updating prompt template to: 'You are a knowledgeable AI. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-a9f54ac19af5' is now available for clients.
[Algo] Received Result: rollout_id='ro-a9f54ac19af5' input='Explain why the sky appears blue using principles of light scattering in 100 words.' start_time=1760451941.7314656
end_time=1760451950.8041675 mode='train' resources_id='rs-b8409d5465c9' status='succeeded' config=RolloutConfig(timeout_seconds=None, unresponsive_seconds=None, max_attempts=1,
retry_condition=[]) metadata={}
[LLM] Span eb949129f1ed43aa (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.streaming': False,
'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a knowledgeable AI. Explain why the sky appears blue using principles of light scattering in 100 words.', 'gen_ai.response.id':
'chatcmpl-CQaGBFeZujxxskZUm2MeW1fNsiKe4', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 150,
'gen_ai.usage.prompt_tokens': 29, 'gen_ai.usage.completion_tokens': 121, 'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false,
"severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false,
"severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant',
'gen_ai.completion.0.content': "The sky appears blue due to Rayleigh scattering, which occurs when sunlight interacts with Earth's atmosphere. Sunlight is composed of various colors, each with
different wavelengths. Shorter wavelengths, like blue and violet, scatter more efficiently than longer wavelengths such as red and yellow. When sunlight passes through the atmosphere, these
shorter blue wavelengths are scattered in all directions by molecules and tiny particles. As a result, when we look up, our eyes perceive this scattered blue light from all parts of the sky,
making it appear predominantly blue. The violet light is mostly absorbed or is less visible, reinforcing the blue appearance.", 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0,
"content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual":
{"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}}]'}
[LLM] Span 2094150c39ec94be (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.temperature': 0.0,
'gen_ai.request.streaming': False, 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': "Evaluate how well the output fulfills the task.\nTask: Explain why the sky appears blue using
principles of light scattering in 100 words.\nOutput: The sky appears blue due to Rayleigh scattering, which occurs when sunlight interacts with Earth's atmosphere. Sunlight is composed of
various colors, each with different wavelengths. Shorter wavelengths, like blue and violet, scatter more efficiently than longer wavelengths such as red and yellow. When sunlight passes through
the atmosphere, these shorter blue wavelengths are scattered in all directions by molecules and tiny particles. As a result, when we look up, our eyes perceive this scattered blue light from all
parts of the sky, making it appear predominantly blue. The violet light is mostly absorbed or is less visible, reinforcing the blue appearance.\nYou must be very critical and strict in your
evaluation.\nReturn only a number between 0 and 1. No text, punctuation, or explanation.", 'gen_ai.response.id': 'chatcmpl-CQaGEAwc6qHLNkC0vblf4vdlaefbq', 'gen_ai.response.model':
'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 190, 'gen_ai.usage.prompt_tokens': 186, 'gen_ai.usage.completion_tokens': 4,
'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false, "severity": "safe"}, "protected_material_code": {"filtered": false,
"detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"},
"violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant', 'gen_ai.completion.0.content': '0.95', 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0,
"content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak": {"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual":
{"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}}]'}
[Algo] Final reward: 0.95

[Algo] Updating prompt template to: 'You are a friendly chatbot. {any_question}'
[Algo] Queuing task for clients...
[Algo] Task 'ro-c67eaa9016b6' is now available for clients.
[Algo] Received Result: rollout_id='ro-c67eaa9016b6' input='Explain why the sky appears blue using principles of light scattering in 100 words.' start_time=1760451950.8512948
end_time=1760451959.8638017 mode='train' resources_id='rs-faf4fa4fa8c9' status='succeeded' config=RolloutConfig(timeout_seconds=None, unresponsive_seconds=None, max_attempts=1,
retry_condition=[]) metadata={}
[LLM] Span 31f4a701f8190eec (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.streaming': False,
'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': 'You are a friendly chatbot. Explain why the sky appears blue using principles of light scattering in 100 words.', 'gen_ai.response.id':
'chatcmpl-CQaGKCAOu3BxLNqPmeTid46yl83fV', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 136,
'gen_ai.usage.prompt_tokens': 29, 'gen_ai.usage.completion_tokens': 107, 'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false,
"severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false,
"severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant',
'gen_ai.completion.0.content': "The sky looks blue because of a process called Rayleigh scattering. When sunlight enters Earth's atmosphere, it is made up of many colors, each with different
wavelengths. Blue light has a shorter wavelength and is scattered in all directions by tiny molecules in the air, like nitrogen and oxygen. This scattered blue light spreads across the sky,
making it appear blue to our eyes. Meanwhile, colors with longer wavelengths, like red and orange, are less scattered and pass through more directly. That's why during the day, the sky
predominantly looks blue!", 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak": {"filtered":
false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}}]'}
[LLM] Span 09a2f2e1eb97977e (openai.chat.completion): {'gen_ai.request.type': 'chat', 'gen_ai.system': 'OpenAI', 'gen_ai.request.model': 'gpt-4.1-nano', 'gen_ai.request.temperature': 0.0,
'gen_ai.request.streaming': False, 'gen_ai.prompt.0.role': 'user', 'gen_ai.prompt.0.content': "Evaluate how well the output fulfills the task.\nTask: Explain why the sky appears blue using
principles of light scattering in 100 words.\nOutput: The sky looks blue because of a process called Rayleigh scattering. When sunlight enters Earth's atmosphere, it is made up of many colors,
each with different wavelengths. Blue light has a shorter wavelength and is scattered in all directions by tiny molecules in the air, like nitrogen and oxygen. This scattered blue light spreads
across the sky, making it appear blue to our eyes. Meanwhile, colors with longer wavelengths, like red and orange, are less scattered and pass through more directly. That's why during the day,
the sky predominantly looks blue!\nYou must be very critical and strict in your evaluation.\nReturn only a number between 0 and 1. No text, punctuation, or explanation.", 'gen_ai.response.id':
'chatcmpl-CQaGNrMwxICsIDVWJxgcVez85lO6w', 'gen_ai.response.model': 'gpt-4.1-nano-2025-04-14', 'gen_ai.openai.system_fingerprint': 'fp_03e44fcc34', 'gen_ai.usage.total_tokens': 174,
'gen_ai.usage.prompt_tokens': 172, 'gen_ai.usage.completion_tokens': 2, 'gen_ai.completion.0.finish_reason': 'stop', 'gen_ai.completion.0.content_filter_results': '{"hate": {"filtered": false,
"severity": "safe"}, "protected_material_code": {"filtered": false, "detected": false}, "protected_material_text": {"filtered": false, "detected": false}, "self_harm": {"filtered": false,
"severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity": "safe"}}', 'gen_ai.completion.0.role': 'assistant',
'gen_ai.completion.0.content': '1', 'gen_ai.prompt.prompt_filter_results': '[{"prompt_index": 0, "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}, "jailbreak":
{"filtered": false, "detected": false}, "self_harm": {"filtered": false, "severity": "safe"}, "sexual": {"filtered": false, "severity": "safe"}, "violence": {"filtered": false, "severity":
"safe"}}}]'}
[Algo] Final reward: 1.0

[Algo] All prompts and their rewards: [('You are a helpful assistant. {any_question}', 0.95), ('You are a knowledgeable AI. {any_question}', 0.95), ('You are a friendly chatbot. {any_question}',
1.0)]
[Algo] Best prompt found: 'You are a friendly chatbot. {any_question}' with reward 1.0
```

Running `python apo_custom_algorithm.py runner` will produce the following output:

```text
2025-10-14 22:23:41,339 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] Setting up tracer...
2025-10-14 22:23:41,343 [INFO] (Process-355341 agentlightning.instrumentation.agentops)   Patched newer version of agentops using handle_chat_attributes
2025-10-14 22:23:41,343 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] Instrumentation applied.
2025-10-14 22:23:41,343 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] Env var set: AGENTOPS_API_KEY=dummy
2025-10-14 22:23:41,343 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] Env var set: AGENTOPS_API_ENDPOINT=http://localhost:56329
2025-10-14 22:23:41,343 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] Env var set: AGENTOPS_APP_URL=http://localhost:56329/notavailable
2025-10-14 22:23:41,343 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] Env var set: AGENTOPS_EXPORTER_ENDPOINT=http://localhost:56329/traces
🖇 AgentOps: [OPENAI INSTRUMENTOR] Error setting up OpenAI streaming wrappers: No module named 'openai.resources.beta.chat'
🖇 AgentOps: Session Replay for default trace: http://localhost:56329/notavailable/sessions?trace_id=606a11562502dbaacfb689da1cd2dd8a
2025-10-14 22:23:41,494 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] AgentOps client initialized.
2025-10-14 22:23:41,494 [INFO] (Process-355341 agentlightning.runner.agent)   [Worker 0] Started async rollouts (max: unlimited).
127.0.0.1 - - [14/Oct/2025 22:23:41] "POST /v3/auth/token HTTP/1.1" 200 -
127.0.0.1 - - [14/Oct/2025 22:25:42] "POST /traces HTTP/1.1" 200 -
[Rollout] LLM returned: The sky appears blue due to Rayleigh scattering, which occurs when sunlight interacts with Earth's atmosphere. Sunlight is composed of various colors, each with different
wavelengths. Shorter wavelengths, like blue and violet, scatter more efficiently than longer wavelengths such as red and yellow. When sunlight passes through the atmosphere, these shorter blue
wavelengths are scattered in all directions by molecules and tiny particles. As a result, when we look up, our eyes perceive this scattered blue light from all parts of the sky, making it appear
predominantly blue. The violet light is mostly absorbed or is less visible, reinforcing the blue appearance.
127.0.0.1 - - [14/Oct/2025 22:25:49] "POST /traces HTTP/1.1" 200 -
[Judge] Judge returned score: 0.95
2025-10-14 22:25:50,803 [INFO] (Process-355341 agentlightning.runner.agent)   [Worker 0 | Rollout ro-a9f54ac19af5] Completed in 4.24s. Collected 3 span(s). Final reward: 0.95
127.0.0.1 - - [14/Oct/2025 22:25:51] "POST /traces HTTP/1.1" 200 -
[Rollout] LLM returned: The sky looks blue because of a process called Rayleigh scattering. When sunlight enters Earth's atmosphere, it is made up of many colors, each with different wavelengths.
Blue light has a shorter wavelength and is scattered in all directions by tiny molecules in the air, like nitrogen and oxygen. This scattered blue light spreads across the sky, making it appear
blue to our eyes. Meanwhile, colors with longer wavelengths, like red and orange, are less scattered and pass through more directly. That's why during the day, the sky predominantly looks blue!
127.0.0.1 - - [14/Oct/2025 22:25:58] "POST /traces HTTP/1.1" 200 -
[Judge] Judge returned score: 1.0
2025-10-14 22:25:59,863 [INFO] (Process-355341 agentlightning.runner.agent)   [Worker 0 | Rollout ro-c67eaa9016b6] Completed in 4.06s. Collected 3 span(s). Final reward: 1.0
127.0.0.1 - - [14/Oct/2025 22:26:00] "POST /traces HTTP/1.1" 200 -
^C2025-10-14 22:30:36,750 [INFO] (Process-355341 agentlightning.instrumentation.agentops)   Unpatched newer version of agentops using handle_chat_attributes
2025-10-14 22:30:36,750 [INFO] (Process-355341 agentlightning.tracer.agentops)   [Worker 0] Instrumentation removed.
2025-10-14 22:30:36,750 [INFO] (Process-355341 agentlightning.instrumentation.agentops)   Stopping AgentOps local server (PID: 355349)...
2025-10-14 22:30:36,754 [INFO] (Process-355341 agentlightning.instrumentation.agentops)   AgentOps local server stopped.
2025-10-14 22:30:36,754 [INFO] (Process-355341 agentlightning.tracer.agentops)   AgentOps server stopped.
```

Store server (`agl store`) log looks like:

```text
2025-10-14 22:25:36,751 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48554 - "GET /get_resources_by_id/rs-a2f1df8ba79f HTTP/1.1" 200 in 0.19 ms
2025-10-14 22:25:37,682 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.95 ms
2025-10-14 22:25:38,694 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.89 ms
2025-10-14 22:25:39,251 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48570 - "GET /get_next_span_sequence_id/ro-de8496b948d0/at-d1c29aa9 HTTP/1.1" 200 in 0.25 ms
2025-10-14 22:25:39,253 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48570 - "POST /add_span HTTP/1.1" 200 in 0.51 ms
2025-10-14 22:25:39,707 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.81 ms
2025-10-14 22:25:40,720 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.91 ms
2025-10-14 22:25:41,561 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48570 - "GET /get_next_span_sequence_id/ro-de8496b948d0/at-d1c29aa9 HTTP/1.1" 200 in 0.18 ms
2025-10-14 22:25:41,561 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48570 - "POST /add_span HTTP/1.1" 200 in 0.33 ms
2025-10-14 22:25:41,562 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48554 - "GET /get_next_span_sequence_id/ro-de8496b948d0/at-d1c29aa9 HTTP/1.1" 200 in 0.10 ms
2025-10-14 22:25:41,563 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48554 - "POST /add_span HTTP/1.1" 200 in 0.23 ms
2025-10-14 22:25:41,563 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48554 - "POST /update_attempt HTTP/1.1" 200 in 0.25 ms
2025-10-14 22:25:41,563 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48554 - "GET /dequeue_rollout HTTP/1.1" 200 in 0.07 ms
2025-10-14 22:25:41,722 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 0.57 ms
2025-10-14 22:25:41,724 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "GET /query_spans/ro-de8496b948d0 HTTP/1.1" 200 in 0.43 ms
2025-10-14 22:25:41,730 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /add_resources HTTP/1.1" 200 in 0.53 ms
2025-10-14 22:25:41,731 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /enqueue_rollout HTTP/1.1" 200 in 0.37 ms
2025-10-14 22:25:41,742 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.64 ms
2025-10-14 22:25:42,755 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.81 ms
2025-10-14 22:25:43,768 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.80 ms
2025-10-14 22:25:44,781 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.90 ms
2025-10-14 22:25:45,793 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.68 ms
2025-10-14 22:25:46,565 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /dequeue_rollout HTTP/1.1" 200 in 0.25 ms
2025-10-14 22:25:46,566 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "POST /update_attempt HTTP/1.1" 200 in 0.28 ms
2025-10-14 22:25:46,566 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /get_resources_by_id/rs-b8409d5465c9 HTTP/1.1" 200 in 0.13 ms
2025-10-14 22:25:46,806 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 11.02 ms
2025-10-14 22:25:47,819 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.93 ms
2025-10-14 22:25:48,831 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.75 ms
2025-10-14 22:25:49,160 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50416 - "GET /get_next_span_sequence_id/ro-a9f54ac19af5/at-2efbb334 HTTP/1.1" 200 in 0.17 ms
2025-10-14 22:25:49,161 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50416 - "POST /add_span HTTP/1.1" 200 in 0.35 ms
2025-10-14 22:25:49,843 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.71 ms
2025-10-14 22:25:50,800 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50416 - "GET /get_next_span_sequence_id/ro-a9f54ac19af5/at-2efbb334 HTTP/1.1" 200 in 0.17 ms
2025-10-14 22:25:50,801 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50416 - "POST /add_span HTTP/1.1" 200 in 0.35 ms
2025-10-14 22:25:50,802 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /get_next_span_sequence_id/ro-a9f54ac19af5/at-2efbb334 HTTP/1.1" 200 in 0.12 ms
2025-10-14 22:25:50,803 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "POST /add_span HTTP/1.1" 200 in 0.27 ms
2025-10-14 22:25:50,804 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "POST /update_attempt HTTP/1.1" 200 in 0.40 ms
2025-10-14 22:25:50,804 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /dequeue_rollout HTTP/1.1" 200 in 0.08 ms
2025-10-14 22:25:50,846 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 0.41 ms
2025-10-14 22:25:50,847 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "GET /query_spans/ro-a9f54ac19af5 HTTP/1.1" 200 in 0.25 ms
2025-10-14 22:25:50,850 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /add_resources HTTP/1.1" 200 in 0.24 ms
2025-10-14 22:25:50,851 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /enqueue_rollout HTTP/1.1" 200 in 0.23 ms
2025-10-14 22:25:50,862 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.58 ms
2025-10-14 22:25:51,875 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.71 ms
2025-10-14 22:25:52,887 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.74 ms
2025-10-14 22:25:53,900 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.83 ms
2025-10-14 22:25:54,913 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.89 ms
2025-10-14 22:25:55,805 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /dequeue_rollout HTTP/1.1" 200 in 0.23 ms
2025-10-14 22:25:55,806 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "POST /update_attempt HTTP/1.1" 200 in 0.33 ms
2025-10-14 22:25:55,807 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /get_resources_by_id/rs-faf4fa4fa8c9 HTTP/1.1" 200 in 0.16 ms
2025-10-14 22:25:55,926 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.83 ms
2025-10-14 22:25:56,938 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.68 ms
2025-10-14 22:25:57,951 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 11.05 ms
2025-10-14 22:25:58,067 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:33776 - "GET /get_next_span_sequence_id/ro-c67eaa9016b6/at-bd941bdb HTTP/1.1" 200 in 0.22 ms
2025-10-14 22:25:58,068 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:33776 - "POST /add_span HTTP/1.1" 200 in 0.46 ms
2025-10-14 22:25:58,964 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 10.54 ms
2025-10-14 22:25:59,860 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:33776 - "GET /get_next_span_sequence_id/ro-c67eaa9016b6/at-bd941bdb HTTP/1.1" 200 in 0.21 ms
2025-10-14 22:25:59,861 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:33776 - "POST /add_span HTTP/1.1" 200 in 0.37 ms
2025-10-14 22:25:59,862 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /get_next_span_sequence_id/ro-c67eaa9016b6/at-bd941bdb HTTP/1.1" 200 in 0.11 ms
2025-10-14 22:25:59,863 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "POST /add_span HTTP/1.1" 200 in 0.26 ms
2025-10-14 22:25:59,863 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "POST /update_attempt HTTP/1.1" 200 in 0.28 ms
2025-10-14 22:25:59,864 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:50404 - "GET /dequeue_rollout HTTP/1.1" 200 in 0.09 ms
2025-10-14 22:25:59,966 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "POST /wait_for_rollouts HTTP/1.1" 200 in 0.29 ms
2025-10-14 22:25:59,967 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:48552 - "GET /query_spans/ro-c67eaa9016b6 HTTP/1.1" 200 in 0.23 ms
2025-10-14 22:26:04,866 [INFO] (Process-352747 agentlightning.store.client_server)   127.0.0.1:32918 - "GET /dequeue_rollout HTTP/1.1" 200 in 0.22 ms
```
