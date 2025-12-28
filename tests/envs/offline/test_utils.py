"""
Tests for torchtrade.envs.offline.utils module.
"""

import pandas as pd
import pytest

from torchtrade.envs.offline.utils import (
    TimeFrame,
    TimeFrameUnit,
    tf_to_timedelta,
    get_timeframe_unit,
    compute_periods_per_year_crypto,
)


class TestTimeFrameUnit:
    """Tests for TimeFrameUnit enum."""

    def test_minute_value(self):
        """Minute unit should have 'Min' as pandas freq value."""
        assert TimeFrameUnit.Minute.value == "Min"

    def test_hour_value(self):
        """Hour unit should have 'H' as pandas freq value."""
        assert TimeFrameUnit.Hour.value == "H"

    def test_day_value(self):
        """Day unit should have 'D' as pandas freq value."""
        assert TimeFrameUnit.Day.value == "D"


class TestTimeFrame:
    """Tests for TimeFrame class."""

    def test_initialization(self):
        """TimeFrame should store value and unit correctly."""
        tf = TimeFrame(5, TimeFrameUnit.Minute)
        assert tf.value == 5
        assert tf.unit == TimeFrameUnit.Minute

    def test_to_pandas_freq_minute(self):
        """to_pandas_freq should return correct string for minutes."""
        tf = TimeFrame(5, TimeFrameUnit.Minute)
        assert tf.to_pandas_freq() == "5Min"

    def test_to_pandas_freq_hour(self):
        """to_pandas_freq should return correct string for hours."""
        tf = TimeFrame(1, TimeFrameUnit.Hour)
        assert tf.to_pandas_freq() == "1H"

    def test_to_pandas_freq_day(self):
        """to_pandas_freq should return correct string for days."""
        tf = TimeFrame(7, TimeFrameUnit.Day)
        assert tf.to_pandas_freq() == "7D"

    def test_obs_key_freq_minute(self):
        """obs_key_freq should return correct observation key for minutes."""
        tf = TimeFrame(15, TimeFrameUnit.Minute)
        assert tf.obs_key_freq() == "15Minute"

    def test_obs_key_freq_hour(self):
        """obs_key_freq should return correct observation key for hours."""
        tf = TimeFrame(4, TimeFrameUnit.Hour)
        assert tf.obs_key_freq() == "4Hour"

    def test_obs_key_freq_day(self):
        """obs_key_freq should return correct observation key for days."""
        tf = TimeFrame(1, TimeFrameUnit.Day)
        assert tf.obs_key_freq() == "1Day"

    def test_various_values(self):
        """TimeFrame should work with various integer values."""
        test_cases = [
            (1, TimeFrameUnit.Minute, "1Min", "1Minute"),
            (5, TimeFrameUnit.Minute, "5Min", "5Minute"),
            (15, TimeFrameUnit.Minute, "15Min", "15Minute"),
            (30, TimeFrameUnit.Minute, "30Min", "30Minute"),
            (1, TimeFrameUnit.Hour, "1H", "1Hour"),
            (4, TimeFrameUnit.Hour, "4H", "4Hour"),
            (24, TimeFrameUnit.Hour, "24H", "24Hour"),
            (1, TimeFrameUnit.Day, "1D", "1Day"),
            (7, TimeFrameUnit.Day, "7D", "7Day"),
        ]
        for value, unit, expected_freq, expected_key in test_cases:
            tf = TimeFrame(value, unit)
            assert tf.to_pandas_freq() == expected_freq
            assert tf.obs_key_freq() == expected_key


class TestTfToTimedelta:
    """Tests for tf_to_timedelta function."""

    def test_minute_conversion(self):
        """Should convert minute timeframe to correct timedelta."""
        tf = TimeFrame(5, TimeFrameUnit.Minute)
        result = tf_to_timedelta(tf)
        assert result == pd.Timedelta(minutes=5)

    def test_hour_conversion(self):
        """Should convert hour timeframe to correct timedelta."""
        tf = TimeFrame(2, TimeFrameUnit.Hour)
        result = tf_to_timedelta(tf)
        assert result == pd.Timedelta(hours=2)

    def test_day_conversion(self):
        """Should convert day timeframe to correct timedelta."""
        tf = TimeFrame(3, TimeFrameUnit.Day)
        result = tf_to_timedelta(tf)
        assert result == pd.Timedelta(days=3)

    def test_timedelta_arithmetic(self):
        """Converted timedelta should work in arithmetic operations."""
        tf = TimeFrame(30, TimeFrameUnit.Minute)
        td = tf_to_timedelta(tf)

        # 30 minutes * 2 = 1 hour
        assert td * 2 == pd.Timedelta(hours=1)

    def test_various_values(self):
        """tf_to_timedelta should work with various timeframe values."""
        test_cases = [
            (1, TimeFrameUnit.Minute, pd.Timedelta(minutes=1)),
            (15, TimeFrameUnit.Minute, pd.Timedelta(minutes=15)),
            (60, TimeFrameUnit.Minute, pd.Timedelta(hours=1)),
            (1, TimeFrameUnit.Hour, pd.Timedelta(hours=1)),
            (24, TimeFrameUnit.Hour, pd.Timedelta(days=1)),
            (1, TimeFrameUnit.Day, pd.Timedelta(days=1)),
            (7, TimeFrameUnit.Day, pd.Timedelta(weeks=1)),
        ]
        for value, unit, expected in test_cases:
            tf = TimeFrame(value, unit)
            assert tf_to_timedelta(tf) == expected


class TestGetTimeframeUnit:
    """Tests for get_timeframe_unit function."""

    def test_minute_variations(self):
        """Should recognize various minute string formats."""
        assert get_timeframe_unit("Min") == TimeFrameUnit.Minute
        assert get_timeframe_unit("min") == TimeFrameUnit.Minute
        assert get_timeframe_unit("Minute") == TimeFrameUnit.Minute

    def test_hour_variations(self):
        """Should recognize various hour string formats."""
        assert get_timeframe_unit("Hour") == TimeFrameUnit.Hour
        assert get_timeframe_unit("hour") == TimeFrameUnit.Hour
        assert get_timeframe_unit("H") == TimeFrameUnit.Hour
        assert get_timeframe_unit("h") == TimeFrameUnit.Hour

    def test_day_variations(self):
        """Should recognize various day string formats."""
        assert get_timeframe_unit("Day") == TimeFrameUnit.Day
        assert get_timeframe_unit("day") == TimeFrameUnit.Day
        assert get_timeframe_unit("D") == TimeFrameUnit.Day

    def test_unknown_unit_raises(self):
        """Should raise ValueError for unknown unit strings."""
        with pytest.raises(ValueError, match="Unknown TimeFrameUnit"):
            get_timeframe_unit("Week")

        with pytest.raises(ValueError, match="Unknown TimeFrameUnit"):
            get_timeframe_unit("invalid")


class TestComputePeriodsPerYearCrypto:
    """Tests for compute_periods_per_year_crypto function."""

    def test_seconds(self):
        """Should compute correct periods for seconds."""
        # 1 second intervals: 365 * 24 * 60 * 60 = 31,536,000 periods
        result = compute_periods_per_year_crypto("S", 1)
        assert result == 365 * 24 * 60 * 60

    def test_minutes(self):
        """Should compute correct periods for minutes."""
        # 1 minute intervals: 365 * 24 * 60 = 525,600 periods
        result = compute_periods_per_year_crypto("Min", 1)
        assert result == 365 * 24 * 60

        # 5 minute intervals
        result = compute_periods_per_year_crypto("Min", 5)
        assert result == (365 * 24 * 60) / 5

    def test_hours(self):
        """Should compute correct periods for hours."""
        # 1 hour intervals: 365 * 24 = 8,760 periods
        result = compute_periods_per_year_crypto("H", 1)
        assert result == 365 * 24

        # 4 hour intervals
        result = compute_periods_per_year_crypto("H", 4)
        assert result == (365 * 24) / 4

    def test_days(self):
        """Should compute correct periods for days."""
        # 1 day intervals: 365 periods
        result = compute_periods_per_year_crypto("D", 1)
        assert result == 365

        # 7 day (weekly) intervals
        result = compute_periods_per_year_crypto("D", 7)
        assert result == 365 / 7

    def test_unknown_unit_raises(self):
        """Should raise ValueError for unknown unit."""
        with pytest.raises(ValueError, match="Unknown execute_on_unit"):
            compute_periods_per_year_crypto("W", 1)

        with pytest.raises(ValueError, match="Unknown execute_on_unit"):
            compute_periods_per_year_crypto("invalid", 1)

    def test_sharpe_ratio_annualization(self):
        """Periods per year should be usable for Sharpe ratio annualization."""
        import math

        # For 1-minute data, annualization factor is sqrt(525600)
        periods = compute_periods_per_year_crypto("Min", 1)
        annualization_factor = math.sqrt(periods)
        assert annualization_factor == pytest.approx(math.sqrt(525600))

        # For daily data, annualization factor is sqrt(365)
        periods = compute_periods_per_year_crypto("D", 1)
        annualization_factor = math.sqrt(periods)
        assert annualization_factor == pytest.approx(math.sqrt(365))
