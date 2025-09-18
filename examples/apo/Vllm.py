# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base import LLM_Model
from vllm import LLM, SamplingParams
from typing import List, Union, Optional
import os
import logging

class VllmModel(LLM_Model):
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 256,
        stop: str = '',
        repetition_penalty: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the VLLM model.

        Args:
            model_path (Optional[str]): Path to the model. Defaults to None.
            max_tokens (int): Maximum number of tokens to generate. Defaults to 256.
            stop (str): Stop sequence for generation. Defaults to ''.
            repetition_penalty (float): Penalty for repetition. Defaults to 1.2.
            logger (Optional[logging.Logger]): Logger object for logging messages.
        """
        self.logger = logger

        # Initialize the VLLM model
        self.llm = LLM(model=model_path)
        self.max_tokens = max_tokens
        self.stop = stop
        self.repetition_penalty = repetition_penalty

    def inference(
        self,
        prompt: Union[str, List[str]],
        use_batch_acceleration: bool = True,
        desc: str = '',
    ) -> Union[str, List[str]]:
        """
        Perform inference using the VLLM model.

        Args:
            prompt (Union[str, List[str]]): Input prompt(s) for the model.
            use_batch_acceleration (bool): Whether to use batch acceleration. Defaults to True.
            desc (str): Description of the inference task for logging.

        Returns:
            Union[str, List[str]]: Generated output(s) from the model.
        """
        # Log the inference call
        if self.logger:
            self.logger.info(f"VLLM | {desc}")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0, 
            repetition_penalty=self.repetition_penalty,
            top_p=0.1,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        
        if use_batch_acceleration and isinstance(prompt, list):
            batch_size = 512
            gen_output_list = []

            for start_idx in range(0, len(prompt), batch_size):
                end_idx = start_idx + batch_size
                sub_gen_input_list = prompt[start_idx:end_idx]
                sub_gen_output_list = self.llm.generate(sub_gen_input_list, sampling_params, use_tqdm=False)
                gen_output_list.extend(sub_gen_output_list)

            return [item.outputs[0].text for item in gen_output_list]

        elif not use_batch_acceleration and isinstance(prompt, str):
            output = self.llm.generate(prompt, sampling_params, use_tqdm=False)
            return output[0].outputs[0].text

if __name__ == "__main__":
    llm = VllmModel("/home/aiscuser/Phi-3-mini-4k-instruct", max_tokens=512, stop='\n\n', repetition_penalty=1.0)
    print(llm.inference("Hello, how are you?",  use_batch_acceleration=False))