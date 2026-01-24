from openai import OpenAI
from dotenv import dotenv_values
import re
import torch
from torch import nn
from torchrl.modules import Actor
from torchrl.data import Categorical, Composite, Unbounded


class _LLMModule(nn.Module):
    """Internal module that wraps LLM API calls for TorchRL compatibility."""

    def __init__(
        self,
        market_data_keys,
        account_state,
        model="gpt-5-nano",
        debug=False,
        feature_keys=None,
        execute_on=None,
        symbol=None,
        action_dict=None,
        initial_action=None,
    ):
        super().__init__()

        # Set defaults if not provided
        if feature_keys is None:
            self.feature_keys = ["close", "open", "high", "low", "volume"]

        if execute_on is None:
            execute_on = "5Minute"
        if symbol is None:
            symbol = "BTC/USD"
        if action_dict is None:
            # Default for fractional action_levels = [0.0, 0.5, 1.0]
            action_dict = {
                "all_cash": 0,      # Target 0% invested
                "half_invested": 1, # Target 50% invested
                "fully_invested": 2 # Target 100% invested
            }

        self.account_state = account_state
        self.market_data_keys = market_data_keys
        self.features_keys = feature_keys
        self.execute_on = execute_on
        self.symbol = symbol
        self.action_dict = action_dict
        self.debug = debug
        self.model = model

        # Track last action for fallback when extraction fails
        # Default to "all_cash" (0% invested) if not specified
        self.last_action = initial_action if initial_action is not None else "all_cash"

        # Generate system prompt dynamically based on action_dict
        available_actions = ", ".join(self.action_dict.keys())
        action_examples = ", ".join([f"<answer>{action}</answer>" for action in self.action_dict.keys()])

        self.system_prompt = f"""
You are a disciplined trading agent for {{symbol}} on the {{execute_on}} timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action from: {available_actions}.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: {action_examples}.
        """

        # Load OpenAI API key from .env
        env = dotenv_values(".env")
        open_ai_key = env.get("OPENAI_API_KEY")
        self.llm = OpenAI(api_key=open_ai_key)

    def construct_prompt(self, *inputs):
        """Construct prompt from market data and account state inputs."""
        # inputs are passed in order of market_data_keys + account_state
        account_state_tensor = inputs[-1]  # Last input is account_state
        market_data_tensors = inputs[:-1]  # All others are market data

        # Construct account state text
        assert account_state_tensor.shape[-1] == len(self.account_state), \
            f"Expected account state shape (..., {len(self.account_state)}), got {account_state_tensor.shape}"

        account_state_text = "Current account state: \n"
        for idx, state in enumerate(self.account_state):
            account_state_text += f"{state}: {round(account_state_tensor[..., idx].item(), 2)}\n"
        account_state_text += "\n---\n"

        # Construct market data text
        market_data_text = "Current market data:\n\n"
        for market_data_key, data_tensor in zip(self.market_data_keys, market_data_tensors):
            data = data_tensor.cpu().numpy().squeeze()  # Shape: (N, 5)
            assert len(data.shape) == 2 and data.shape[1] == len(self.feature_keys), \
                f"Expected market data shape (N, {len(self.feature_keys)}), got {data.shape}"
            N = data.shape[0]

            market_data_text += f"{market_data_key}:\n\n"
            header = " | " + " | ".join([f"{key:>8}" for key in self.feature_keys])
            market_data_text += header + "\n\n"

            for t in range(N):
                row = data[t]
                market_data_text += f"{row[1]:8.1f} | {row[1]:8.1f} | {row[2]:8.1f} | {row[3]:8.1f} | {row[4]:8.1f}\n"

            market_data_text += "\n"

        return account_state_text + market_data_text

    def extract_action(self, response):
        """Extract action from LLM response."""
        answer_pattern = "<answer>(.*?)</answer>"
        match = re.search(answer_pattern, response)
        if match:
            action = match.group(1)
            # Update last_action for future fallback
            self.last_action = action
            return action
        else:
            import warnings
            warnings.warn(
                f"Failed to extract action from LLM response. "
                f"Maintaining previous position: {self.last_action}"
            )
            return self.last_action

    def forward(self, *inputs):
        """Forward pass: generate LLM response and extract action + thinking."""
        prompt = self.construct_prompt(*inputs)

        if self.debug:
            print("SYSTEM PROMPT:\n")
            print(self.system_prompt)
            print("\nPROMPT:\n")
            print(prompt)

        # Query LLM
        response = self.llm.responses.create(
            model=self.model,
            instructions=self.system_prompt.format(symbol=self.symbol, execute_on=self.execute_on),
            input=prompt,
        )
        response_text = response.output_text

        if self.debug:
            print("\nRESPONSE:\n")
            print(response_text)

        # Extract action
        action_name = self.extract_action(response_text)
        action_idx = self.action_dict[action_name]

        # Return action as tensor and full response as thinking trace
        action_tensor = torch.tensor([action_idx], dtype=torch.long)

        return action_tensor, response_text


class LLMActor(Actor):
    """
    LLM-based trading actor that uses language models to make trading decisions.

    This actor inherits from TorchRL's Actor class and wraps LLM API calls for trading.
    It constructs prompts from market data and account state, queries an LLM, and extracts
    trading actions from the model's response. The actor expects responses in a structured
    format with reasoning enclosed in <think></think> tags and final actions in
    <answer></answer> tags.

    The actor outputs two keys to the TensorDict:
    - "action": The trading action index (int)
    - "thinking": The full LLM response text including reasoning

    Parameters
    ----------
    market_data_keys : list of str
        **Required.** List of TensorDict keys containing market data observations.
        Each key should correspond to a multi-timeframe OHLCV tensor in the format
        "market_data_{timeframe}_{window_size}" (e.g., ["market_data_1Minute_12",
        "market_data_5Minute_8"]). The actor will construct prompts using all specified
        market data keys.

        **Note:** This can be obtained directly from any TorchTrade environment instance
        via `env.market_data_keys`.

    account_state : list of str
        **Required.** List of account state variable names that define the structure of the
        account_state tensor. The order must match the account_state tensor dimensions in the
        TensorDict. Common examples include ["cash", "position_size", "position_value",
        "entry_price", "current_price", "unrealized_pnlpct", "holding_time"] for long-only
        environments, or with additional futures-specific fields like "leverage", "margin_ratio",
        "liquidation_price" for futures environments.

        **Note:** This can be obtained directly from any TorchTrade environment instance
        via `env.account_state`.

    model : str, optional
        OpenAI model identifier to use for action generation. Default is "gpt-5-nano".

    debug : bool, optional
        If True, prints the system prompt, constructed prompt, and LLM response to stdout
        for debugging purposes. Default is False.

    feature_keys : list of str, optional
        List of feature column names present in the market data tensors. Default is
        ["close", "open", "high", "low", "volume"]. Currently used for documentation
        purposes in the code but can be extended for feature-specific processing.

    execute_on : str, optional
        The primary timeframe on which trading decisions are executed. Used in the system
        prompt to inform the LLM about the decision frequency. Default is "5Minute".
        Examples: "1Minute", "5Minute", "15Minute", "1Hour".

    symbol : str, optional
        Trading symbol/pair that the agent is trading. Used in the system prompt to provide
        market context to the LLM. Default is "BTC/USD". Examples: "ETH/USD", "AAPL".

    action_dict : dict, optional
        Mapping from action names (str) to action indices (int) that defines the available
        actions and their numerical encodings. Default is {"all_cash": 0, "half_invested": 1,
        "fully_invested": 2} for fractional long-only environments with action_levels = [0.0, 0.5, 1.0].
        The keys determine what actions the LLM can choose from, and the values specify
        the corresponding action indices that will be set in the output TensorDict.
        Custom action spaces can be defined for different trading strategies
        (e.g., {"close": 0, "hold": 1, "buy_sl1_tp1": 2, "buy_sl2_tp2": 3}).

    initial_action : str, optional
        Initial action to use as fallback when LLM response extraction fails. Default is "all_cash"
        (0% invested). This action is maintained whenever the LLM fails to provide a valid response
        in the expected <answer></answer> format.

    Examples
    --------
    Basic usage with required parameters:

    >>> actor = LLMActor(
    ...     market_data_keys=["market_data_1Minute_12", "market_data_5Minute_8"],
    ...     account_state=["cash", "position_size", "position_value", "entry_price",
    ...                    "current_price", "unrealized_pnlpct", "holding_time"]
    ... )

    Using environment attributes to configure the actor:

    >>> from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
    >>> config = SeqLongOnlyEnvConfig(...)
    >>> env = SeqLongOnlyEnv(config)
    >>> actor = LLMActor(
    ...     market_data_keys=env.market_data_keys,
    ...     account_state=env.account_state,
    ...     symbol=config.symbol,
    ...     execute_on=config.execute_on
    ... )

    Custom configuration for ETH trading with custom fractional levels:

    >>> actor = LLMActor(
    ...     market_data_keys=["market_data_1Minute_20"],
    ...     account_state=["cash", "position_size", "position_value", "entry_price",
    ...                    "current_price", "unrealized_pnlpct", "holding_time"],
    ...     symbol="ETH/USD",
    ...     execute_on="1Minute",
    ...     action_dict={"all_cash": 0, "quarter_invested": 1, "half_invested": 2, "fully_invested": 3},
    ...     debug=True
    ... )

    Using the actor with a TensorDict:

    >>> output_td = actor(tensordict)  # Returns tensordict with "action" and "thinking" keys
    """

    def __init__(
        self,
        market_data_keys,
        account_state,
        model="gpt-5-nano",
        debug=False,
        feature_keys=None,
        execute_on=None,
        symbol=None,
        action_dict=None,
        initial_action=None,
    ):
        # Set default action_dict if not provided
        if action_dict is None:
            # Default for fractional action_levels = [0.0, 0.5, 1.0]
            action_dict = {
                "all_cash": 0,      # Target 0% invested
                "half_invested": 1, # Target 50% invested
                "fully_invested": 2 # Target 100% invested
            }

        # Create the internal LLM module
        module = _LLMModule(
            market_data_keys=market_data_keys,
            account_state=account_state,
            model=model,
            debug=debug,
            feature_keys=feature_keys,
            execute_on=execute_on,
            symbol=symbol,
            action_dict=action_dict,
            initial_action=initial_action,
        )

        # Define input keys: market data keys + account_state
        in_keys = list(market_data_keys) + ["account_state"]

        # Define output keys: action and thinking trace
        out_keys = ["action", "thinking"]

        # Create output specs for TorchRL collector compatibility
        # Action spec: Categorical with n actions
        n_actions = len(action_dict)
        action_spec = Categorical(n_actions, shape=torch.Size([1]))

        # Thinking spec: Unbounded as placeholder
        # (actual output will be a string, which TorchDict handles gracefully)
        thinking_spec = Unbounded(shape=torch.Size([]))

        # Combine into Composite spec
        spec = Composite({
            "action": action_spec,
            "thinking": thinking_spec,
        }, shape=torch.Size([]))

        # Initialize Actor parent class
        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=out_keys,
            spec=spec,
        )