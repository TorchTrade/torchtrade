from openai import OpenAI
from dotenv import dotenv_values   # <-- returns a dict
import re


class LLMActor():
    """
    LLM-based trading actor that uses language models to make trading decisions.

    This actor constructs prompts from market data and account state, queries an LLM,
    and extracts trading actions from the model's response. The actor expects responses
    in a structured format with reasoning enclosed in <think></think> tags and final
    actions in <answer></answer> tags.

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

    model : str, optional
        OpenAI model identifier to use for action generation. Default is "gpt-5-nano".

    debug : bool, optional
        If True, prints the system prompt, constructed prompt, and LLM response to stdout
        for debugging purposes. Default is False.

    account_state : list of str
        **Required.** List of account state variable names that define the structure of the
        account_state tensor. The order must match the account_state tensor dimensions in the
        TensorDict. Common examples include ["cash", "position_size", "position_value",
        "entry_price", "current_price", "unrealized_pnlpct", "holding_time"] for long-only
        environments, or with additional futures-specific fields like "leverage", "margin_ratio",
        "liquidation_price" for futures environments.

        **Note:** This can be obtained directly from any TorchTrade environment instance
        via `env.account_state`.

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
        actions and their numerical encodings. Default is {"buy": 2, "sell": 0, "hold": 1}.
        The keys determine what actions the LLM can choose from, and the values specify
        the corresponding action indices that will be set in the output TensorDict.
        Custom action spaces can be defined for different trading strategies
        (e.g., {"close": 0, "hold": 1, "buy_sl1_tp1": 2, "buy_sl2_tp2": 3}).

    Attributes
    ----------
    llm : OpenAI
        OpenAI client instance initialized with API key from .env file.

    system_prompt : str
        Dynamically generated system prompt that instructs the LLM on available actions,
        response format, and trading context (symbol, timeframe).

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

    Custom configuration for ETH trading with SLTP actions:

    >>> actor = LLMActor(
    ...     market_data_keys=["market_data_1Minute_20"],
    ...     account_state=["cash", "position_size", "position_value", "entry_price",
    ...                    "current_price", "unrealized_pnlpct", "holding_time"],
    ...     symbol="ETH/USD",
    ...     execute_on="1Minute",
    ...     action_dict={"hold": 0, "close": 1, "buy_conservative": 2, "buy_aggressive": 3},
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
        action_dict=None
    ):
        super().__init__()

        # Set defaults if not provided
        if feature_keys is None:
            feature_keys = ["close", "open", "high", "low", "volume"]
        if execute_on is None:
            execute_on = "5Minute"
        if symbol is None:
            symbol = "BTC/USD"
        if action_dict is None:
            action_dict = {"buy": 2, "sell": 0, "hold": 1}

        self.account_state = account_state
        self.market_data_keys = market_data_keys
        self.features_keys = feature_keys
        self.execute_on = execute_on
        self.symbol = symbol
        self.action_dict = action_dict
        self.debug = debug
        self.model = model

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

        # Load *all* variables from .env into a dict
        env = dotenv_values(".env")        # you can give a different path if needed

        # Grab just the one you care about
        open_ai_key = env.get("OPENAI_API_KEY")
        self.llm = OpenAI(
                api_key=open_ai_key
            )
    def construct_prompt(self, tensordict):
        account_state = self.construct_account_state(tensordict)
        market_data = self.construct_market_data(tensordict)
        return account_state + market_data

    def construct_account_state(self, tensordict):
        account_state = tensordict.get("account_state")
        assert account_state.shape == (1, len(self.account_state)), f"Expected account state shape (1, {len(self.account_state)}), got {account_state.shape}"
        out = """Current account state: \n"""
        for idx, state in enumerate(self.account_state):
            out += f"{state}: {round(account_state[0, idx].item(), 2)}\n"
        out += "\n---\n"
        return out

    def construct_market_data(self, tensordict):
        """
        Example output:

        Current market data:

        market_data_1Minute_12:

        close |     open |     high |      low |   volume

        101950.9 | 101950.9 | 101980.5 | 101936.0 |      0.0
        101977.0 | 101977.0 | 102026.7 | 101963.2 |      0.0
        102017.6 | 102017.6 | 102092.8 | 101953.7 |      0.0
        101963.3 | 101963.3 | 102027.3 | 101963.3 |      0.0
        102028.9 | 102028.9 | 102068.4 | 102000.9 |      0.0
        102045.4 | 102045.4 | 102045.4 | 101974.2 |      0.0
        101986.1 | 101986.1 | 102013.8 | 101954.7 |      0.0
        101985.4 | 101985.4 | 102028.0 | 101877.9 |      0.0
        101921.4 | 101921.4 | 101941.5 | 101898.6 |      0.0
        101931.1 | 101931.1 | 101931.1 | 101828.4 |      0.0
        101875.6 | 101875.6 | 101875.6 | 101694.1 |      0.0
        101707.6 | 101707.6 | 101804.4 | 101707.6 |      0.0

        market_data_5Minute_8:

        close |     open |     high |      low |   volume

        101986.0 | 101986.0 | 102018.0 | 101944.7 |      0.0
        101979.9 | 101979.9 | 102035.2 | 101950.3 |      0.0
        ...

        """
        out = "Current market data:\n\n"

        for market_data_key in self.market_data_keys:
            data = tensordict[market_data_key].numpy().squeeze()  # Shape: (N, 5)
            assert len(data.shape) == 2 and data.shape[1] == 5, f"Expected market data shape (N, 5), got {data.shape}"
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
        

    def extract_action(self, response):
        answer_pattern = "<answer>(.*?)</answer>"
        match = re.search(answer_pattern, response)
        if match:
            return match.group(1)
        else:
            print("No answer found in response")
            return "hold"

    def generate(self, prompt):
        response = self.llm.responses.create(
            model=self.model,
            instructions=self.system_prompt.format(symbol=self.symbol, execute_on=self.execute_on),
            input=prompt,
        )
        
        return response.output_text

    def __call__(self, tensordict):
        return self.forward(tensordict)
    
    def forward(self, tensordict):
        prompt = self.construct_prompt(tensordict)
        if self.debug:
            print("SYSTEM PROMPT:\n")
            print(self.system_prompt)
            print("PROMPT:\n")
            print(prompt)
        response = self.generate(prompt)
        if self.debug:
            print("RESPONSE:\n")
            print(response)
        action = self.extract_action(response)
        float_action = self.action_dict[action]
        tensordict.set("action", float_action)
        # Add thinking
        thinking_pattern = "<think>(.*?)</think>"
        match = re.search(thinking_pattern, response)
        if match:
            thinking = match.group(1)
            tensordict.set("thinking", thinking)
        return tensordict