"""
Mean Reversion Actor using Bollinger Bands and Stochastic RSI.

This module implements a mean reversion trading strategy that works best in
ranging/sideways markets.
"""

from torchtrade.actor.rulebased.meanreversion.actor import MeanReversionActor

__all__ = ["MeanReversionActor"]
