import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px


class HumanActor():
    def __init__(self, symbol, features, market_data_keys, account_state_key, action_spec):
        super().__init__()

        self.symbol = symbol
        self.features = features
        if self.features[0].startswith("features_"):
            self.features = [feat.split("features_")[1] for feat in self.features]
        self.market_data_keys = market_data_keys
        self.account_state_key = account_state_key
        self.account_state = ["cash", "portfolio_value", "position_size", "entry_price", "unrealized_pnlpct", "holding_time"]

        self.action_spec = action_spec
        self.action_dict = {"buy": 2, "sell": 0, "hold": 1}


    def construct_account_state(self, tensordict):
        account_state = tensordict.get(self.account_state_key)
        assert account_state.shape == (1, 6), f"Expected account state shape (1, 6), got {account_state.shape}"
        out = """Current account state: \n"""
        for idx, state in enumerate(self.account_state):
            out += f"{state}: {round(account_state[0, idx].item(), 2)}\n"
        out += "\n---\n"
        return out


    def __call__(self, tensordict):
        return self.forward(tensordict)
    
    def forward(self, tensordict):
        # visualize market data
        data = {key.split("market_data_")[1]: tensordict.get(key) for key in self.market_data_keys}
        dfs = {key: pd.DataFrame(value.cpu().squeeze(), columns=self.features) for key, value in data.items()}
        combined_dashboard(dfs)
        # output accout state
        account_state = self.construct_account_state(tensordict)
        print(account_state)

        print("\nWhat action do you want to take? (buy, sell, hold)")
        action = input()
        float_action = self.action_dict[action]
        tensordict.set("action", [float_action])

        return tensordict


def combined_dashboard(dfs):
    names = list(dfs.keys())
    n = len(names)

    # Get unique column names from the first dataframe
    cols = dfs[names[0]].columns

    # Assign consistent colors using Plotly Express default color sequence
    color_map = {col: px.colors.qualitative.Plotly[i % 10] for i, col in enumerate(cols)}

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=names
    )

    for i, (name, df) in enumerate(dfs.items(), start=1):
        for col in df.columns:
            show = (i == 1)  # show legend only for first subplot

            fig.add_trace(
                go.Scatter(
                    y=df[col],
                    mode="lines",
                    name=col,
                    showlegend=show,
                    legendgroup=col,
                    line=dict(color=color_map[col])  # <-- consistent color
                ),
                row=i,
                col=1
            )

    fig.update_layout(
        height=300 * n,
        showlegend=True
    )

    fig.show()