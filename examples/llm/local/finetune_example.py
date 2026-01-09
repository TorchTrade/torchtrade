"""
Fine-tuning example for LocalLLMActor with unsloth.

This example demonstrates how to fine-tune a local LLM on trading data
to improve its decision-making abilities.

The fine-tuning process uses:
- LoRA (Low-Rank Adaptation) for efficient training
- 4-bit quantization to reduce memory usage
- Trading-specific prompt-response pairs

Requirements:
    pip install unsloth
    pip install torch transformers datasets

Dataset format:
    The training data should be a list of dictionaries with:
    - "prompt": The market data and account state
    - "response": The correct action and reasoning
"""

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import pandas as pd
import numpy as np


def create_training_data():
    """
    Create synthetic training data for demonstration.

    In practice, you would:
    1. Collect historical market data
    2. Run a profitable baseline strategy
    3. Generate prompt-response pairs from successful trades
    4. Or use human expert annotations
    """
    training_examples = [
        {
            "instruction": """You are a disciplined trading agent for BTC/USD on the 5Minute timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.""",
            "input": """Current account state:
cash: 10000.0
position_size: 0.0
position_value: 0.0
entry_price: 0.0
current_price: 50000.0
unrealized_pnlpct: 0.0

---
Current market data:

market_data_5Minute_8:

  close |     open |     high |      low |   volume

 49800.0 |  49900.0 |  49950.0 |  49750.0 |     10.5
 49900.0 |  49800.0 |  50000.0 |  49800.0 |     12.3
 50000.0 |  49900.0 |  50100.0 |  49900.0 |     15.8
 50100.0 |  50000.0 |  50200.0 |  50000.0 |     18.2
""",
            "output": """<think>
Looking at the market data:
- Price has been trending upward from 49800 to 50100 over the last 4 periods
- Volume is increasing (10.5 → 18.2), confirming the uptrend
- Each period is making higher highs and higher lows
- We have no position currently and sufficient cash

This shows a strong bullish momentum with increasing volume support.
A buy entry here could capture continuation of the trend.
</think>

<answer>buy</answer>""",
        },
        {
            "instruction": """You are a disciplined trading agent for BTC/USD on the 5Minute timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.""",
            "input": """Current account state:
cash: 5000.0
position_size: 0.1
position_value: 5000.0
entry_price: 50000.0
current_price: 49000.0
unrealized_pnlpct: -2.0

---
Current market data:

market_data_5Minute_8:

  close |     open |     high |      low |   volume

 50000.0 |  50100.0 |  50150.0 |  49900.0 |     15.2
 49800.0 |  50000.0 |  50050.0 |  49700.0 |     20.5
 49500.0 |  49800.0 |  49850.0 |  49400.0 |     25.8
 49000.0 |  49500.0 |  49600.0 |  48900.0 |     32.1
""",
            "output": """<think>
Analyzing the situation:
- We entered at 50000, now price is at 49000 (-2% unrealized loss)
- Price has been declining steadily: 50000 → 49800 → 49500 → 49000
- Volume is increasing during the decline (15.2 → 32.1), indicating strong selling pressure
- Each period shows lower highs and lower lows

This is a clear downtrend with accelerating momentum. Our position is losing money
and the trend shows no signs of reversal. We should exit to prevent further losses.
</think>

<answer>sell</answer>""",
        },
        {
            "instruction": """You are a disciplined trading agent for BTC/USD on the 5Minute timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.""",
            "input": """Current account state:
cash: 4000.0
position_size: 0.12
position_value: 6000.0
entry_price: 48000.0
current_price: 50000.0
unrealized_pnlpct: 4.17

---
Current market data:

market_data_5Minute_8:

  close |     open |     high |      low |   volume

 49900.0 |  49850.0 |  50050.0 |  49800.0 |     12.5
 50000.0 |  49900.0 |  50100.0 |  49850.0 |     11.8
 50050.0 |  50000.0 |  50150.0 |  49950.0 |     10.2
 50000.0 |  50050.0 |  50100.0 |  49950.0 |      9.5
""",
            "output": """<think>
Reviewing our position:
- We bought at 48000, current price is 50000 (+4.17% unrealized profit)
- Price is consolidating around 50000 with small fluctuations
- Volume is decreasing (12.5 → 9.5), showing reduced conviction
- Price is making neither clear higher highs nor lower lows

We're in profit but the market is sideways with declining volume.
This suggests the uptrend may be exhausted. However, there's no clear reversal signal yet.
Best to hold and wait for either a breakout continuation or a clear reversal pattern.
</think>

<answer>hold</answer>""",
        },
    ]

    return training_examples


def format_prompt(example):
    """Format training example into the prompt format expected by the model."""
    return f"""{example['instruction']}

{example['input']}

{example['output']}"""


def main():
    """Run the fine-tuning process."""
    print("=" * 80)
    print("Local LLM Trading Actor - Fine-tuning Example")
    print("=" * 80)
    print()

    # Configuration
    model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    max_seq_length = 2048
    output_dir = "./qwen3_trading_finetuned"

    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print()

    # Load model
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("Model loaded successfully!")
    print()

    # Configure LoRA
    print("Configuring LoRA for efficient fine-tuning...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("LoRA configuration complete!")
    print()

    # Prepare dataset
    print("Creating training dataset...")
    training_data = create_training_data()

    # Format for training
    formatted_data = [
        {
            "text": format_prompt(example)
        }
        for example in training_data
    ]

    dataset = Dataset.from_list(formatted_data)
    print(f"Dataset created with {len(dataset)} examples")
    print()

    # Training arguments
    print("Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=5,
        save_strategy="epoch",
        report_to="none",  # Change to "wandb" if you want to log to Weights & Biases
    )
    print("Training configuration complete!")
    print()

    # Create trainer
    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )
    print("Trainer created!")
    print()

    # Train
    print("=" * 80)
    print("Starting fine-tuning...")
    print("=" * 80)
    print()

    trainer.train()

    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print()

    # Save model
    print(f"Saving fine-tuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")
    print()

    print("=" * 80)
    print("Next steps:")
    print("=" * 80)
    print(f"1. Load your fine-tuned model:")
    print(f"   actor = LocalLLMActor(model_name='{output_dir}')")
    print()
    print("2. Test it on live/backtesting environments")
    print()
    print("3. Collect more data from successful trades and iterate!")
    print()


if __name__ == "__main__":
    main()
