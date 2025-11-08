from openai import OpenAI
from dotenv import dotenv_values   # <-- returns a dict
import re


class LLMActor():
    def __init__(self, model="gpt-5-nano", debug=False):
        super().__init__()
        self.system_prompt ="""
You are a disciplined trading agent for {symbol} on the {execute_on} timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.
        """
        self.account_state = ["cash", "portfolio_value", "position_size", "entry_price", "unrealized_pnlpct", "holding_time"]
        self.market_data_keys = ["market_data_1Minute_12", "market_data_5Minute_8", "market_data_15Minute_8", "market_data_1Hour_24"]
        self.features_keys = ["close", "open", "high", "low", "volume"]
        self.execute_on = "5Minute"
        self.symbol = "BTC/USD"
        self.action_dict = {"buy": 2, "sell": 0, "hold": 1}
        self.debug = debug
        self.model = model

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
        assert account_state.shape == (1, 6), f"Expected account state shape (1, 6), got {account_state.shape}"
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
        return tensordict