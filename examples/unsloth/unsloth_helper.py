# Copyright (c) Microsoft. All rights reserved.

"""Helper module for Unsloth training with LoRA.

This module provides utilities for training language models using the Unsloth library,
which optimizes training performance with 4-bit quantization and LoRA (Low-Rank Adaptation).

The training function should be run in a separate process to ensure GPU memory is properly freed.
"""

from datasets import Dataset as HuggingFaceDataset  # type: ignore
from rich.console import Console

console = Console()


def unsloth_training(model_path: str, sft_dataset: HuggingFaceDataset, next_model_path: str):
    """Train a Unsloth model on a SFT dataset.

    This is recommended to be run in a separate process to avoid GPU memory issues.

    Args:
        model_path: The path to the model to train. Must be a local path.
        sft_dataset: The SFT dataset to train on.
        next_model_path: The path to save the trained model.
    """

    from unsloth import FastLanguageModel

    # The two imports must come in this order to make unsloth patch work.
    if True:
        # The SFTTrainer is actually patched by unsloth.
        from trl import SFTConfig, SFTTrainer  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore
        model_name=model_path,
        max_seq_length=4096,  # Choose any for long context!
        load_in_4bit=True,  # 4 bit quantization to reduce memory
    )

    # Config the model to use LoRA
    model = FastLanguageModel.get_peft_model(  # type: ignore
        model,  # type: ignore
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    sft_config = (
        SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            warmup_steps=5,
            max_steps=60,  # Maximum number of steps to train for
            # num_train_epochs = 1, # Set this for 1 full training run
            learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            # FIXME: For some reason, report_to="none" still tries to report to W&B when it's installed.
            report_to="none",  # Use this for W&B etc
        ),
    )

    trainer = SFTTrainer(
        model=model,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
        train_dataset=sft_dataset,
        args=sft_config,
    )

    trainer_stats = trainer.train()  # type: ignore
    console.print(f"[bold red][Algo][/bold red] Trainer stats: {trainer_stats}")

    # Save in 16-bit for vLLM inference later
    model.save_pretrained_merged(next_model_path, tokenizer, save_method="merged_16bit")  # type: ignore

    # All unsloth memory should be freed now.
    # But this won't happen unless you run unsloth_training in a separate process!
