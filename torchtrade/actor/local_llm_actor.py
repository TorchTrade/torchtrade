"""Local LLM Actor for trading using vllm or transformers backends."""
import re
from typing import Dict, Optional, Tuple, Union
import torch


class LocalLLMActor:
    """
    Local LLM-based trading actor with vllm or transformers backend.

    Similar to LLMActor but uses local models instead of OpenAI API.
    Supports multiple action spaces (standard 3-action, SLTP, futures_sltp).

    Args:
        model: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        backend: Inference backend ("vllm" or "transformers")
        device: Device for inference ("cuda", "cpu", "mps")
        quantization: Quantization mode (None, "4bit", "8bit")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        debug: Enable debug output
        action_space_type: Type of action space ("standard", "sltp", "futures_sltp")
        action_map: Action map for SLTP environments (dict[int, tuple])
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        backend: str = "vllm",
        device: str = "cuda",
        quantization: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        debug: bool = False,
        action_space_type: str = "standard",
        action_map: Optional[Dict[int, Tuple[Optional[str], Optional[float], Optional[float]]]] = None,
    ):
        super().__init__()

        self.model_name = model
        self.backend = backend
        self.device = device
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.debug = debug
        self.action_space_type = action_space_type
        self.action_map = action_map

        # Standard system prompt for 3-action envs
        self.system_prompt_standard = """
You are a disciplined trading agent for {symbol} on the {execute_on} timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.
        """

        # Futures system prompt for 10-element account state
        self.system_prompt_futures = """
You are a futures trading agent for {symbol} on the {execute_on} timeframe with {leverage}x leverage.
At each step, you receive account state (including margin and liquidation info) and market data.
{action_instructions}

- Consider liquidation risk when position_size != 0
- Leverage amplifies both gains and losses
- Think step-by-step inside <think></think>.
- Output your final action: <answer>{action_format}</answer>
        """

        # Account state definitions
        self.account_state_standard = [
            "cash", "position_size", "position_value", "entry_price",
            "current_price", "unrealized_pnlpct", "holding_time"
        ]

        self.account_state_futures = [
            "cash", "position_size", "position_value", "entry_price",
            "current_price", "unrealized_pnl_pct", "leverage",
            "margin_ratio", "liquidation_price", "holding_time"
        ]

        # Market data keys (will be auto-detected from tensordict)
        self.market_data_keys = []
        self.features_keys = ["close", "open", "high", "low", "volume"]

        # Default env-specific attributes (will be set from first tensordict)
        self.execute_on = "5Minute"
        self.symbol = "BTC/USD"

        # Action mappings
        if action_space_type == "standard":
            self.action_dict = {"buy": 2, "sell": 0, "hold": 1}
        elif action_space_type in ["sltp", "futures_sltp"]:
            # For SLTP, we'll use numeric actions
            if action_map is None:
                raise ValueError("action_map required for SLTP action spaces")
            self.action_dict = None  # Will extract numeric action directly

        # Initialize LLM backend
        self.llm = None
        self.tokenizer = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize vllm or transformers backend."""
        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams

                # Configure vllm
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=["</answer>"],  # Stop at answer tag
                )

                # Initialize LLM
                kwargs = {
                    "model": self.model_name,
                    "trust_remote_code": True,
                }

                # Add quantization if specified
                if self.quantization == "4bit":
                    kwargs["quantization"] = "bitsandbytes"
                    kwargs["load_format"] = "bitsandbytes"
                elif self.quantization == "8bit":
                    kwargs["quantization"] = "bitsandbytes_8bit"

                # Add GPU memory utilization
                kwargs["gpu_memory_utilization"] = 0.9

                self.llm = LLM(**kwargs)

                if self.debug:
                    print(f"[LocalLLMActor] Initialized vllm with model: {self.model_name}")

            except ImportError:
                print("[LocalLLMActor] vllm not available, falling back to transformers")
                self.backend = "transformers"
                self._initialize_llm()

        elif self.backend == "transformers":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                # Configure model loading
                model_kwargs = {
                    "trust_remote_code": True,
                }

                # Add quantization if specified
                if self.quantization == "4bit":
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif self.quantization == "8bit":
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                else:
                    # No quantization - use device
                    if self.device == "cuda" and torch.cuda.is_available():
                        model_kwargs["device_map"] = "auto"
                    else:
                        model_kwargs["device_map"] = self.device

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )

                # Create pipeline
                self.llm = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                )

                if self.debug:
                    print(f"[LocalLLMActor] Initialized transformers with model: {self.model_name}")

            except ImportError as e:
                raise ImportError(
                    "Neither vllm nor transformers available. "
                    "Install with: pip install 'torchtrade[llm_local]'"
                ) from e
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'vllm' or 'transformers'")

    def _build_action_space_description(self) -> Tuple[str, str]:
        """
        Build action space description and format for prompt.

        Returns:
            (action_instructions, action_format) tuple
        """
        if self.action_space_type == "standard":
            instructions = "You must choose exactly one action: buy, sell, or hold."
            format_str = "buy/sell/hold"
        elif self.action_space_type in ["sltp", "futures_sltp"]:
            instructions = "Available actions:\n"
            for idx, (side, sl, tp) in self.action_map.items():
                if side is None:
                    instructions += f"  {idx}: Hold/Close position\n"
                else:
                    instructions += f"  {idx}: {side.capitalize()} SL={sl*100:.1f}% TP={tp*100:.1f}%\n"
            instructions += "\nYou must choose exactly one action by number (e.g., 0, 1, 2, ...)."
            format_str = "action_number"
        else:
            raise ValueError(f"Unknown action_space_type: {self.action_space_type}")

        return instructions, format_str

    def construct_prompt(self, tensordict):
        """Construct full prompt from tensordict."""
        account_state = self.construct_account_state(tensordict)
        market_data = self.construct_market_data(tensordict)
        return account_state + market_data

    def construct_account_state(self, tensordict):
        """
        Construct account state section from tensordict.

        Auto-detects 7-element (standard) vs 10-element (futures) account state.
        """
        account_state = tensordict.get("account_state")

        # Auto-detect account state size
        if account_state.dim() == 2:
            state_size = account_state.shape[1]
        else:
            state_size = account_state.shape[0]

        if state_size == 7:
            state_labels = self.account_state_standard
        elif state_size == 10:
            state_labels = self.account_state_futures
        else:
            raise ValueError(f"Unknown account state size: {state_size}. Expected 7 or 10.")

        # Flatten if needed
        if account_state.dim() == 2:
            account_state = account_state.squeeze(0)

        out = "Current account state:\n"
        for idx, label in enumerate(state_labels):
            out += f"{label}: {round(account_state[idx].item(), 4)}\n"
        out += "\n---\n"

        return out

    def construct_market_data(self, tensordict):
        """
        Construct market data table from tensordict.

        Reuses the pattern from LLMActor.
        """
        out = "Current market data:\n\n"

        # Auto-detect market data keys from tensordict
        if not self.market_data_keys:
            for key in tensordict.keys():
                if key.startswith("market_data_"):
                    self.market_data_keys.append(key)

        for market_data_key in self.market_data_keys:
            if market_data_key not in tensordict:
                continue

            data = tensordict[market_data_key].cpu().numpy()

            # Handle different tensor shapes
            if data.ndim == 3:
                data = data.squeeze(0)  # Remove batch dim

            if data.ndim != 2 or data.shape[1] != 5:
                if self.debug:
                    print(f"[Warning] Unexpected market data shape for {market_data_key}: {data.shape}")
                continue

            N = data.shape[0]

            # Header
            out += f"{market_data_key}:\n\n"

            # Table header
            header = f" {'close':>7} | {'open':>8} | {'high':>8} | {'low':>8} | {'volume':>8}"
            out += header + "\n\n"

            # Rows (OHLCV format: close, open, high, low, volume)
            for t in range(N):
                row = data[t]
                out += f"{row[0]:8.1f} | {row[1]:8.1f} | {row[2]:8.1f} | {row[3]:8.1f} | {row[4]:8.1f}\n"

            out += "\n"

        return out

    def _format_system_prompt(self, tensordict) -> str:
        """Format system prompt based on environment type."""
        # Detect account state size to choose prompt
        account_state = tensordict.get("account_state")
        if account_state.dim() == 2:
            state_size = account_state.shape[1]
        else:
            state_size = account_state.shape[0]

        # Build action instructions
        action_instructions, action_format = self._build_action_space_description()

        if state_size == 10:
            # Futures environment
            # Extract leverage from account state
            if account_state.dim() == 2:
                leverage = account_state[0, 6].item()
            else:
                leverage = account_state[6].item()

            return self.system_prompt_futures.format(
                symbol=self.symbol,
                execute_on=self.execute_on,
                leverage=int(leverage) if leverage > 0 else 1,
                action_instructions=action_instructions,
                action_format=action_format,
            )
        else:
            # Standard environment
            if self.action_space_type in ["sltp", "futures_sltp"]:
                # SLTP with standard account state
                return self.system_prompt_futures.format(
                    symbol=self.symbol,
                    execute_on=self.execute_on,
                    leverage=1,
                    action_instructions=action_instructions,
                    action_format=action_format,
                )
            else:
                return self.system_prompt_standard.format(
                    symbol=self.symbol,
                    execute_on=self.execute_on,
                )

    def extract_action(self, response: str) -> str:
        """Extract action from <answer> tags."""
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            action = match.group(1).strip()
            return action
        else:
            if self.debug:
                print("[Warning] No <answer> tag found in response, defaulting to hold/0")
            return "hold" if self.action_space_type == "standard" else "0"

    def extract_thinking(self, response: str) -> Optional[str]:
        """Extract thinking from <think> tags."""
        thinking_pattern = r"<think>(.*?)</think>"
        match = re.search(thinking_pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def map_action_to_index(self, action_str: str) -> int:
        """Map action string to index."""
        if self.action_space_type == "standard":
            # Map buy/sell/hold to index
            action_lower = action_str.lower()
            if action_lower in self.action_dict:
                return self.action_dict[action_lower]
            else:
                if self.debug:
                    print(f"[Warning] Unknown action '{action_str}', defaulting to hold")
                return 1  # hold
        elif self.action_space_type in ["sltp", "futures_sltp"]:
            # Parse numeric action
            try:
                action_idx = int(action_str)
                if action_idx in self.action_map:
                    return action_idx
                else:
                    if self.debug:
                        print(f"[Warning] Invalid action {action_idx}, defaulting to 0")
                    return 0  # hold/close
            except ValueError:
                if self.debug:
                    print(f"[Warning] Could not parse action '{action_str}', defaulting to 0")
                return 0
        else:
            raise ValueError(f"Unknown action_space_type: {self.action_space_type}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using vllm or transformers backend."""
        if self.backend == "vllm":
            from vllm import SamplingParams

            # Format prompt for chat models (using chat template if available)
            try:
                # Try to use tokenizer's chat template
                if hasattr(self.llm, "get_tokenizer"):
                    tokenizer = self.llm.get_tokenizer()
                    if hasattr(tokenizer, "apply_chat_template"):
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        # Fallback: simple concatenation
                        prompt = f"{system_prompt}\n\n{user_prompt}"
                else:
                    prompt = f"{system_prompt}\n\n{user_prompt}"
            except:
                # Fallback: simple concatenation
                prompt = f"{system_prompt}\n\n{user_prompt}"

            # Generate
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text

            return response

        elif self.backend == "transformers":
            # Format as chat if possible
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = f"{system_prompt}\n\n{user_prompt}"

            # Generate
            outputs = self.llm(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                return_full_text=False,
            )

            response = outputs[0]["generated_text"]
            return response
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def __call__(self, tensordict):
        """Standard actor interface."""
        return self.forward(tensordict)

    def forward(self, tensordict):
        """
        Main forward pass.

        Args:
            tensordict: TensorDict with observation

        Returns:
            tensordict with action set
        """
        # Construct prompts
        system_prompt = self._format_system_prompt(tensordict)
        user_prompt = self.construct_prompt(tensordict)

        if self.debug:
            print("=" * 80)
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("\nUSER PROMPT:")
            print(user_prompt)
            print("=" * 80)

        # Generate response
        response = self.generate(system_prompt, user_prompt)

        if self.debug:
            print("RESPONSE:")
            print(response)
            print("=" * 80)

        # Extract action
        action_str = self.extract_action(response)
        action_idx = self.map_action_to_index(action_str)

        # Set action in tensordict
        tensordict.set("action", torch.tensor(action_idx, dtype=torch.long))

        # Extract and set thinking if present
        thinking = self.extract_thinking(response)
        if thinking:
            tensordict.set("thinking", thinking)

        return tensordict
