from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

# def wrap_method(inst, name, post_fn):
#     old = getattr(inst, name)
#     def new(*args, **kwargs):
#         result = old(*args, **kwargs)
#         return post_fn(result)
#     setattr(inst, name, types.MethodType(new, inst))


# def _convert_encoding(
#         self,
#         encoding: EncodingFast,
#         return_token_type_ids: Optional[bool] = None,
#         return_attention_mask: Optional[bool] = None,
#         return_overflowing_tokens: bool = False,
#         return_special_tokens_mask: bool = False,
#         return_offsets_mapping: bool = False,
#         return_length: bool = False,
#         verbose: bool = True,
#     ) -> tuple[dict[str, Any], list[EncodingFast]]:
#         """
#         Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
#         of encodings, take care of building a batch from overflowing tokens.

#         Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
#         lists (overflows) of lists (tokens).

#         Output shape: (overflows, sequence length)
#         """
#         self.add_special_tokens({'additional_special_tokens': ['<AGL_MESSAGE_START>', '<AGL_MESSAGE_END>']})
#         agl_start_token = self.convert_tokens_to_ids("<AGL_MESSAGE_START>")
#         agl_end_token = self.convert_tokens_to_ids("<AGL_MESSAGE_END>")
#         if return_token_type_ids is None:
#             return_token_type_ids = "token_type_ids" in self.model_input_names
#         if return_attention_mask is None:
#             return_attention_mask = "attention_mask" in self.model_input_names

#         if return_overflowing_tokens and encoding.overflowing is not None:
#             encodings = [encoding] + encoding.overflowing
#         else:
#             encodings = [encoding]

#         encoding_dict = defaultdict(list)
#         for e in encodings:
#             is_zero_sep_token = [id in (agl_start_token, agl_end_token) for id in e.ids]
#             encoding_dict["input_ids"].append([id for id, is_sep in zip(e.ids, is_zero_sep_token) if not is_sep])

#             if return_token_type_ids:
#                 # encoding_dict["token_type_ids"].append(e.type_ids)
#                 encoding_dict["token_type_ids"].append([id for id, is_sep in zip(e.type_ids, is_zero_sep_token) if not is_sep])
#             if return_attention_mask:
#                 # encoding_dict["attention_mask"].append(e.attention_mask)
#                 encoding_dict["attention_mask"].append([mask for mask, is_sep in zip(e.attention_mask, is_zero_sep_token) if not is_sep])
#             if return_special_tokens_mask:
#                 # encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
#                 encoding_dict["special_tokens_mask"].append([mask for mask, is_sep in zip(e.special_tokens_mask, is_zero_sep_token) if not is_sep])
#             if return_offsets_mapping:
#                 # encoding_dict["offset_mapping"].append(e.offsets)
#                 encoding_dict["offset_mapping"].append([offset for offset, is_sep in zip(e.offsets, is_zero_sep_token) if not is_sep])
#             if return_length:
#                 encoding_dict["length"].append(len(e.ids) - sum(is_zero_sep_token))

#         return encoding_dict, encodings

# import pdb; pdb.set_trace()

# tokenizer.add_special_tokens({'additional_special_tokens': ['<ZERO_SEP>']})

# tokenizer.encode("hello")
# # 14990

# tokenizer.encode("he")
# # 383
# tokenizer.encode("llo")
# # 75, 385
# tokenizer.encode("<ZERO_SEP>")
# # 151665

# tokenizer.encode("he<ZERO_SEP>llo")
# # 383, 75, 385

# tokenizer(["!\n\n", "!", "\n\n", "<ZERO_SEP>", "!<ZERO_SEP>\n\n"])
# # {'input_ids': [[2219], [0], [271], [], [0, 271]], 'attention_mask': [[1], [1], [1], [], [1, 1]]}

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, cast
import agentlightning as agl

tokenizer.add_special_tokens({'additional_special_tokens': ['<AGL_MESSAGE_START>', '<AGL_MESSAGE_END>']})

hist_messages: List[Dict[str, Any]] = [{"role": "user", "content": "123"}]
hist_messages = agl.add_message(hist_messages, {"role": "assistant", "content": "456"})

tokenizer.encode('<AGL_MESSAGE_START><AGL_MESSAGE_END>')
# [151665, 151666]
tokenizer.apply_chat_template(hist_messages)
# [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151665, 16, 17, 18, 151666, 151645, 198, 151644, 77091, 198, 151665, 19, 20, 21, 151666, 151645, 198]
tokenizer.decode([151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 16, 17, 18, 151645, 198, 151644, 77091, 198, 19, 20, 21, 151645, 198])
# '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n<AGL_MESSAGE_START>123<AGL_MESSAGE_END><|im_end|>\n<|im_start|>assistant\n<AGL_MESSAGE_START>456<AGL_MESSAGE_END><|im_end|>\n'
# import pdb; pdb.set_trace()

from vllm import LLM, SamplingParams
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen2.5-Coder-3B-Instruct", tokenizer="Qwen/Qwen2.5-Coder-3B-Instruct")

outputs = llm.generate("<AGL_MESSAGE_START>456<AGL_MESSAGE_END>", sampling_params)
print(outputs)
import pdb; pdb.set_trace()
# [RequestOutput(request_id=2, prompt='<AGL_MESSAGE_START>456<AGL_MESSAGE_END>', prompt_token_ids=[27, 1890, 43, 14641, 13044, 29, 19, 20, 21, 27, 1890, 43, 14641, 10898, 29], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='"\n\nIt seems like you are trying to create a message format for a specific system', token_ids=[1837, 2132, 4977, 1075, 498, 525, 4460, 311, 1855, 264, 1943, 3561, 369, 264, 3151, 1849], cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=None, lora_request=None, num_cached_tokens=0, multi_modal_placeholders={})]
