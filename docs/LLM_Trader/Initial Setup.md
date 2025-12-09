# LLM Text Trader

## Initial Setup

Create an LLM Actor class that uses OpenAI's API to generate actions based on the current state of the environment.
The actor generates a prompt based on the current state of the environment.

Currently a simple initial setup to test and get started.

Example Actor
```python
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
        ...
```

And can be used with the torchrl collector as follows:
```python
actor = LLMActor()
collector = SyncDataCollector(
    env,
    actor,
    init_random_frames=0,
    frames_per_batch=1,
    total_frames=total_farming_steps,
    device=device,
)
collector.set_seed(42)
return collector
```

Example with debug=True:
```python
SYSTEM PROMPT:


You are a disciplined trading agent for {symbol} on the {execute_on} timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.
        
        
PROMPT:

Current account state: 
cash: 326.61
portfolio_value: 0.0
position_size: 0.0
entry_price: 0.0
unrealized_pnlpct: 0.0
holding_time: 0.0

---
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
101973.9 | 101973.9 | 102065.6 | 101840.5 |      0.0
101839.3 | 101839.3 | 101945.1 | 101839.3 |      0.0
101885.5 | 101885.5 | 102016.2 | 101885.5 |      0.0
101886.8 | 101886.8 | 102092.8 | 101874.6 |      0.0
102028.9 | 102028.9 | 102068.4 | 101877.9 |      0.0
101931.1 | 101931.1 | 101931.1 | 101694.1 |      0.0

market_data_15Minute_8:

   close |     open |     high |      low |   volume

102553.3 | 102553.3 | 102632.9 | 102414.3 |      0.0
102459.6 | 102459.6 | 102483.4 | 101906.1 |      0.0
101992.7 | 101992.7 | 102068.2 | 101715.4 |      0.0
101823.7 | 101823.7 | 102059.7 | 101682.3 |      0.0
101705.1 | 101705.1 | 102025.7 | 101705.1 |      0.0
101986.0 | 101986.0 | 102065.6 | 101840.5 |      0.0
101839.3 | 101839.3 | 102092.8 | 101839.3 |      0.0
102028.9 | 102028.9 | 102068.4 | 101694.1 |      0.0

market_data_1Hour_24:

   close |     open |     high |      low |   volume

100271.4 | 100271.4 | 101024.1 |  99440.6 |      0.1
100746.2 | 100746.2 | 101267.4 |  99270.4 |      0.0
100745.1 | 100745.1 | 101573.7 | 100378.5 |      0.0
101104.8 | 101104.8 | 102722.5 | 100861.0 |      0.1
102346.4 | 102346.4 | 102889.7 | 102190.9 |      0.2
102549.2 | 102549.2 | 103271.6 | 102042.1 |      0.1
103249.2 | 103249.2 | 104136.0 | 102949.0 |      0.0
103723.9 | 103723.9 | 103871.9 | 103299.2 |      0.0
103860.2 | 103860.2 | 104143.2 | 103533.9 |      0.0
103599.1 | 103599.1 | 103757.3 | 103272.1 |      0.0
103310.8 | 103310.8 | 103364.5 | 102480.4 |      0.0
102618.8 | 102618.8 | 103185.1 | 102589.2 |      0.0
102937.3 | 102937.3 | 103378.7 | 102697.9 |      0.0
102790.7 | 102790.7 | 103355.6 | 102526.0 |      0.0
102558.1 | 102558.1 | 102858.2 | 102277.2 |      0.0
102489.5 | 102489.5 | 102688.1 | 102170.4 |      0.0
102263.4 | 102263.4 | 103179.7 | 102040.8 |      0.1
102180.2 | 102180.2 | 102383.2 | 101832.2 |      0.0
102311.0 | 102311.0 | 102683.2 | 102047.2 |      0.0
102399.0 | 102399.0 | 102599.9 | 102112.5 |      0.0
102335.6 | 102335.6 | 102639.0 | 102238.8 |      0.0
102417.7 | 102417.7 | 102632.9 | 101715.4 |      0.0
101823.7 | 101823.7 | 102092.8 | 101682.3 |      0.0
102028.9 | 102028.9 | 102068.4 | 101694.1 |      0.0


RESPONSE:

<think>Account is flat (no position). Assess multi-timeframe price action focused on 5-min:
- 5-min closes: 101986 → 101980 → 101974 → 101839 → 101885 → 101887 → 102028.9 → 101931.1 — recent net drop, lower high then lower low.
- 1-min shows intraminute weakness with several declining closes down to ~101707.6 low.
- 15-min and 1‑hour show price around ~102k but short-term momentum on the 5-min is bearish.
- Volume fields are 0.0 in provided data (no confirmation), and there is no clear reversal or setup to enter safely.
Given no position and short-term downside momentum without a confirmed entry signal, the disciplined action is to wait for a clearer trade signal (trend continuation confirmation or a defined reversal with volume/price confirmation).</think>
<answer>hold</answer>
```


**NOTE:** Outouts could get stored and used to fine tune a local LLM.

**NOTE:** ADD Transform to fetch news from X or other platform to extend the observation space!