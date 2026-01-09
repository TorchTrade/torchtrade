"""
Local LLM Actor using unsloth for fast inference with quantized models.

This actor uses local language models (e.g., Qwen3-0.6B) for trading decisions,
providing an alternative to cloud-based APIs like OpenAI.
"""

import re
import torch
from typing import Optional


class LocalLLMActor:
    """
    Trading agent using local LLMs via unsloth for fast inference.

    This actor is compatible with 4-bit quantized models and can run on
    resource-constrained devices like Raspberry Pi.

    Args:
        model_name: HuggingFace model identifier (default: unsloth/Qwen3-0.6B-unsloth-bnb-4bit)
        max_seq_length: Maximum sequence length for the model
        load_in_4bit: Whether to load model in 4-bit quantization
        debug: Whether to print debug information
        device: Device to run inference on (default: auto-detect)
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        debug: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.debug = debug

        # Trading configuration
        self.system_prompt = """You are a disciplined trading agent for {symbol} on the {execute_on} timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.
"""

        self.account_state = [
            "cash",
            "position_size",
            "position_value",
            "entry_price",
            "current_price",
            "unrealized_pnlpct",
            "holding_time",
        ]
        self.market_data_keys = [
            "market_data_1Minute_12",
            "market_data_5Minute_8",
            "market_data_15Minute_8",
            "market_data_1Hour_24",
        ]
        self.features_keys = ["close", "open", "high", "low", "volume"]
        self.execute_on = "5Minute"
        self.symbol = "BTC/USD"
        self.action_dict = {"buy": 2, "sell": 0, "hold": 1}

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model using unsloth
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer using unsloth's FastLanguageModel."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "unsloth is required for LocalLLMActor. "
                "Install with: pip install unsloth"
            )

        # Load model with unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.load_in_4bit,
        )

        # Enable fast inference mode
        FastLanguageModel.for_inference(self.model)

        if self.debug:
            print(f"Loaded model: {self.model_name}")
            print(f"Device: {self.device}")
            print(f"4-bit quantization: {self.load_in_4bit}")

    def construct_prompt(self, tensordict):
        """Construct the full prompt from tensordict."""
        account_state = self.construct_account_state(tensordict)
        market_data = self.construct_market_data(tensordict)
        return account_state + market_data

    def construct_account_state(self, tensordict):
        """Format account state into text."""
        account_state = tensordict.get("account_state")
        assert account_state.shape == (1, 6), (
            f"Expected account state shape (1, 6), got {account_state.shape}"
        )

        out = "Current account state:\n"
        for idx, state in enumerate(self.account_state):
            out += f"{state}: {round(account_state[0, idx].item(), 2)}\n"
        out += "\n---\n"
        return out

    def construct_market_data(self, tensordict):
        """Format market data into tabular text."""
        out = "Current market data:\n\n"

        for market_data_key in self.market_data_keys:
            data = tensordict[market_data_key].numpy().squeeze()  # Shape: (N, 5)
            assert len(data.shape) == 2 and data.shape[1] == 5, (
                f"Expected market data shape (N, 5), got {data.shape}"
            )
            N = data.shape[0]

            # Header
            out += f"{market_data_key}:\n\n"

            # Table header
            header = f" {'close':>7} | {'open':>8} | {'high':>8} | {'low':>8} | {'volume':>8}"
            out += header + "\n\n"

            # Rows
            for t in range(N):
                row = data[t]
                out += f"{row[1]:8.1f} | {row[1]:8.1f} | {row[2]:8.1f} | {row[3]:8.1f} | {row[4]:8.1f}\n"

            out += "\n"

        return out

    def extract_action(self, response: str) -> str:
        """Extract action from model response."""
        answer_pattern = "<answer>(.*?)</answer>"
        match = re.search(answer_pattern, response)
        if match:
            return match.group(1).strip().lower()
        else:
            if self.debug:
                print("No answer found in response, defaulting to 'hold'")
            return "hold"

    def generate(self, prompt: str) -> str:
        """Generate response from the local model."""
        # Format with system prompt
        full_prompt = self.system_prompt.format(
            symbol=self.symbol, execute_on=self.execute_on
        ) + "\n\n" + prompt

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response (model echoes it)
        if full_prompt in response:
            response = response.replace(full_prompt, "").strip()

        return response

    def __call__(self, tensordict):
        """Make the actor callable."""
        return self.forward(tensordict)

    def forward(self, tensordict):
        """
        Forward pass: generate action from tensordict.

        Args:
            tensordict: TensorDict containing account_state and market_data_* keys

        Returns:
            tensordict: Input tensordict with 'action' and optionally 'thinking' added
        """
        prompt = self.construct_prompt(tensordict)

        if self.debug:
            print("SYSTEM PROMPT:\n")
            print(self.system_prompt.format(symbol=self.symbol, execute_on=self.execute_on))
            print("\nPROMPT:\n")
            print(prompt)

        response = self.generate(prompt)

        if self.debug:
            print("\nRESPONSE:\n")
            print(response)

        # Extract action
        action = self.extract_action(response)
        action_idx = self.action_dict.get(action, 1)  # Default to hold
        tensordict.set("action", action_idx)

        # Extract thinking if present
        thinking_pattern = "<think>(.*?)</think>"
        match = re.search(thinking_pattern, response, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            tensordict.set("thinking", thinking)

        return tensordict
