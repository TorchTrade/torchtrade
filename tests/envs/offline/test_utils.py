"""
Tests for torchtrade.envs.offline.utils module.
"""

import pandas as pd
import pytest

from torchtrade.envs.offline.utils import (
    TimeFrame,
    TimeFrameUnit,
    tf_to_timedelta,
    compute_periods_per_year_crypto,
    parse_timeframe_string,
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


class TestParseTimeframeString:
    """Tests for parse_timeframe_string function."""

    def test_standard_formats(self):
        """Should parse standard pandas-style formats."""
        result = parse_timeframe_string("5Min")
        assert result.value == 5
        assert result.unit == TimeFrameUnit.Minute

        result = parse_timeframe_string("1Hour")
        assert result.value == 1
        assert result.unit == TimeFrameUnit.Hour

        result = parse_timeframe_string("7Day")
        assert result.value == 7
        assert result.unit == TimeFrameUnit.Day

    def test_case_insensitive(self):
        """Should handle case variations."""
        result = parse_timeframe_string("5min")
        assert result.value == 5
        assert result.unit == TimeFrameUnit.Minute

        result = parse_timeframe_string("5MIN")
        assert result.value == 5
        assert result.unit == TimeFrameUnit.Minute

        result = parse_timeframe_string("1hour")
        assert result.value == 1
        assert result.unit == TimeFrameUnit.Hour

        result = parse_timeframe_string("1HOUR")
        assert result.value == 1
        assert result.unit == TimeFrameUnit.Hour

    def test_with_spaces(self):
        """Should handle spaces between number and unit."""
        result = parse_timeframe_string("5 Min")
        assert result.value == 5
        assert result.unit == TimeFrameUnit.Minute

        result = parse_timeframe_string("1 Hour")
        assert result.value == 1
        assert result.unit == TimeFrameUnit.Hour

        result = parse_timeframe_string("  15  Minute  ")
        assert result.value == 15
        assert result.unit == TimeFrameUnit.Minute

    def test_plural_forms(self):
        """Should parse plural unit names."""
        result = parse_timeframe_string("5Minutes")
        assert result.value == 5
        assert result.unit == TimeFrameUnit.Minute

        result = parse_timeframe_string("1Hours")
        assert result.value == 1
        assert result.unit == TimeFrameUnit.Hour

        result = parse_timeframe_string("7Days")
        assert result.value == 7
        assert result.unit == TimeFrameUnit.Day

    def test_single_letter_abbreviations(self):
        """Should parse single-letter units."""
        result = parse_timeframe_string("5M")
        assert result.value == 5
        assert result.unit == TimeFrameUnit.Minute

        result = parse_timeframe_string("1H")
        assert result.value == 1
        assert result.unit == TimeFrameUnit.Hour

        result = parse_timeframe_string("7D")
        assert result.value == 7
        assert result.unit == TimeFrameUnit.Day

    def test_large_values(self):
        """Should handle large numeric values."""
        result = parse_timeframe_string("999Min")
        assert result.value == 999
        assert result.unit == TimeFrameUnit.Minute

        # 1440Min will trigger a warning (can be normalized to 1day)
        # We suppress it here since we're testing large values, not warning behavior
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = parse_timeframe_string("1440Min")
            assert result.value == 1440
            assert result.unit == TimeFrameUnit.Minute

    def test_invalid_format_raises(self):
        """Should raise ValueError for invalid formats."""
        with pytest.raises(ValueError, match="Invalid timeframe format"):
            parse_timeframe_string("invalid")

        with pytest.raises(ValueError, match="Invalid timeframe format"):
            parse_timeframe_string("Min5")  # reversed

        with pytest.raises(ValueError, match="Invalid timeframe format"):
            parse_timeframe_string("5")  # missing unit

        with pytest.raises(ValueError, match="Invalid timeframe format"):
            parse_timeframe_string("")  # empty

    def test_unknown_unit_raises(self):
        """Should raise ValueError for unknown units."""
        with pytest.raises(ValueError, match="Unknown time unit"):
            parse_timeframe_string("5Week")

        with pytest.raises(ValueError, match="Unknown time unit"):
            parse_timeframe_string("1Month")

    def test_result_compatible_with_timeframe(self):
        """Result should be usable as a TimeFrame object."""
        result = parse_timeframe_string("15Min")

        # Should work with to_pandas_freq
        assert result.to_pandas_freq() == "15Min"

        # Should work with obs_key_freq
        assert result.obs_key_freq() == "15Minute"

        # Should work with tf_to_timedelta
        assert tf_to_timedelta(result) == pd.Timedelta(minutes=15)


class TestTimeframeWarnings:
    """Tests for non-canonical timeframe format warnings (Issue #99)."""

    def test_60min_issues_warning(self):
        """Should warn when using 60min instead of 1hour."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tf = parse_timeframe_string("60min")

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "60min" in str(w[0].message).lower()
            assert "1hour" in str(w[0].message).lower()
            assert "60Minute" in str(w[0].message)
            assert "1Hour" in str(w[0].message)

            # Should still create correct TimeFrame
            assert tf.value == 60
            assert tf.unit == TimeFrameUnit.Minute

    def test_120min_issues_warning(self):
        """Should warn when using 120min instead of 2hours."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tf = parse_timeframe_string("120min")

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "120min" in str(w[0].message).lower()
            assert "2hours" in str(w[0].message).lower()

    def test_24hour_issues_warning(self):
        """Should warn when using 24hour instead of 1day."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tf = parse_timeframe_string("24hour")

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "24hour" in str(w[0].message).lower()
            assert "1day" in str(w[0].message).lower()

    def test_1440min_issues_warning(self):
        """Should warn when using 1440min instead of 1day."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tf = parse_timeframe_string("1440min")

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "1440min" in str(w[0].message).lower()
            assert "1day" in str(w[0].message).lower()

    def test_canonical_forms_no_warning(self):
        """Canonical forms should not issue warnings."""
        import warnings

        canonical_forms = ["1hour", "2hours", "1day", "5min", "15min"]

        for form in canonical_forms:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                parse_timeframe_string(form)

                # Should not issue any warnings
                assert len(w) == 0, f"Unexpected warning for canonical form '{form}': {w[0].message if w else 'none'}"

    def test_non_divisible_no_warning(self):
        """Non-divisible timeframes should not issue warnings."""
        import warnings

        # These cannot be normalized cleanly
        non_divisible = ["5min", "15min", "45min", "3hour", "7hour"]

        for form in non_divisible:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                parse_timeframe_string(form)

                # Should not issue any warnings
                assert len(w) == 0, f"Unexpected warning for non-divisible form '{form}'"


class TestTimeFrameComparison:
    """Tests for TimeFrame comparison operators (Issue #10 lookahead bias fix).

    These tests verify the comparison logic used by the lookahead bias fix
    in sampler.py line 75: `if tf > execute_on:` to determine which
    timeframes need offset adjustment.
    """

    def test_equality_same_value_and_unit(self):
        """TimeFrames with same value and unit should be equal."""
        tf1 = TimeFrame(5, TimeFrameUnit.Minute)
        tf2 = TimeFrame(5, TimeFrameUnit.Minute)

        assert tf1 == tf2
        assert not (tf1 != tf2)

    def test_inequality_different_values(self):
        """TimeFrames with different values should not be equal."""
        tf1 = TimeFrame(5, TimeFrameUnit.Minute)
        tf2 = TimeFrame(10, TimeFrameUnit.Minute)

        assert tf1 != tf2
        assert not (tf1 == tf2)

    def test_inequality_different_units(self):
        """TimeFrames with different units should not be equal, even if same duration."""
        tf_1day = TimeFrame(1, TimeFrameUnit.Day)
        tf_24hours = TimeFrame(24, TimeFrameUnit.Hour)
        tf_1440min = TimeFrame(1440, TimeFrameUnit.Minute)

        # Structural inequality (different representations)
        assert tf_1day != tf_24hours
        assert tf_24hours != tf_1440min
        assert tf_1day != tf_1440min

        # But same temporal duration via to_minutes()
        assert tf_1day.to_minutes() == tf_24hours.to_minutes() == tf_1440min.to_minutes()

    def test_less_than_same_unit(self):
        """Less-than should work correctly for same unit."""
        tf_5min = TimeFrame(5, TimeFrameUnit.Minute)
        tf_10min = TimeFrame(10, TimeFrameUnit.Minute)
        tf_15min = TimeFrame(15, TimeFrameUnit.Minute)

        assert tf_5min < tf_10min < tf_15min
        assert not (tf_10min < tf_5min)

    def test_less_than_different_units(self):
        """Less-than should compare by duration across different units."""
        tf_30min = TimeFrame(30, TimeFrameUnit.Minute)
        tf_1hour = TimeFrame(1, TimeFrameUnit.Hour)
        tf_2hour = TimeFrame(2, TimeFrameUnit.Hour)

        # 30 min < 1 hour < 2 hours
        assert tf_30min < tf_1hour
        assert tf_1hour < tf_2hour
        assert tf_30min < tf_2hour

    def test_less_than_equal_durations(self):
        """TimeFrames with equal durations should not compare as less-than."""
        tf_1day = TimeFrame(1, TimeFrameUnit.Day)
        tf_24hours = TimeFrame(24, TimeFrameUnit.Hour)

        # Equal duration, neither is less than the other
        assert not (tf_1day < tf_24hours)
        assert not (tf_24hours < tf_1day)

    def test_greater_than_same_unit(self):
        """Greater-than should work correctly for same unit."""
        tf_5min = TimeFrame(5, TimeFrameUnit.Minute)
        tf_10min = TimeFrame(10, TimeFrameUnit.Minute)

        assert tf_10min > tf_5min
        assert not (tf_5min > tf_10min)

    def test_greater_than_different_units(self):
        """Greater-than should compare by duration across different units.

        This is the critical comparison used in sampler.py line 75:
        `if tf > execute_on:` to determine which timeframes to offset.
        """
        tf_5min = TimeFrame(5, TimeFrameUnit.Minute)
        tf_1hour = TimeFrame(1, TimeFrameUnit.Hour)
        tf_1day = TimeFrame(1, TimeFrameUnit.Day)

        # Hour > Minute, Day > Hour
        assert tf_1hour > tf_5min
        assert tf_1day > tf_1hour
        assert tf_1day > tf_5min

    def test_sorting_mixed_units(self):
        """TimeFrames with mixed units should sort correctly by duration."""
        tfs = [
            TimeFrame(1, TimeFrameUnit.Day),      # 1440 min
            TimeFrame(1, TimeFrameUnit.Hour),     # 60 min
            TimeFrame(30, TimeFrameUnit.Minute),  # 30 min
            TimeFrame(2, TimeFrameUnit.Hour),     # 120 min
            TimeFrame(15, TimeFrameUnit.Minute),  # 15 min
        ]

        sorted_tfs = sorted(tfs)
        minutes = [tf.to_minutes() for tf in sorted_tfs]

        # Should be sorted: 15, 30, 60, 120, 1440
        assert minutes == [15.0, 30.0, 60.0, 120.0, 1440.0]

    def test_hash_consistency_with_equality(self):
        """Objects that compare equal should have same hash (hash table requirement)."""
        tf1 = TimeFrame(5, TimeFrameUnit.Minute)
        tf2 = TimeFrame(5, TimeFrameUnit.Minute)

        assert tf1 == tf2
        assert hash(tf1) == hash(tf2)

    def test_hash_different_for_unequal(self):
        """Different timeframes should (ideally) have different hashes."""
        tf1 = TimeFrame(5, TimeFrameUnit.Minute)
        tf2 = TimeFrame(10, TimeFrameUnit.Minute)
        tf3 = TimeFrame(5, TimeFrameUnit.Hour)

        # Not guaranteed but very likely
        assert hash(tf1) != hash(tf2)
        assert hash(tf1) != hash(tf3)
        assert hash(tf2) != hash(tf3)

    def test_hashable_in_set(self):
        """TimeFrames should be usable as set/dict keys."""
        tf1 = TimeFrame(5, TimeFrameUnit.Minute)
        tf2 = TimeFrame(5, TimeFrameUnit.Minute)  # Equal to tf1
        tf3 = TimeFrame(10, TimeFrameUnit.Minute)

        # Can create set
        tf_set = {tf1, tf2, tf3}

        # tf1 and tf2 are equal, so set should have 2 items
        assert len(tf_set) == 2

        # Can use as dict key
        tf_dict = {tf1: "five", tf3: "ten"}
        assert tf_dict[tf2] == "five"  # tf2 == tf1

    def test_total_ordering_transitivity(self):
        """Verify @total_ordering maintains transitivity (a < b < c => a < c)."""
        a = TimeFrame(5, TimeFrameUnit.Minute)
        b = TimeFrame(10, TimeFrameUnit.Minute)
        c = TimeFrame(15, TimeFrameUnit.Minute)

        # Transitivity
        assert a < b
        assert b < c
        assert a < c  # Must hold

        # Test all operators derived by @total_ordering
        assert a <= b <= c
        assert c > b > a
        assert c >= b >= a

    def test_to_minutes_conversion_accuracy(self):
        """to_minutes() should return accurate minute counts for all units."""
        test_cases = [
            (TimeFrame(1, TimeFrameUnit.Minute), 1.0),
            (TimeFrame(5, TimeFrameUnit.Minute), 5.0),
            (TimeFrame(60, TimeFrameUnit.Minute), 60.0),
            (TimeFrame(1, TimeFrameUnit.Hour), 60.0),
            (TimeFrame(2, TimeFrameUnit.Hour), 120.0),
            (TimeFrame(24, TimeFrameUnit.Hour), 1440.0),
            (TimeFrame(1, TimeFrameUnit.Day), 1440.0),
            (TimeFrame(7, TimeFrameUnit.Day), 10080.0),
        ]

        for tf, expected_minutes in test_cases:
            assert tf.to_minutes() == expected_minutes

    def test_comparison_used_by_lookahead_fix(self):
        """Verify comparison works as expected by lookahead bias fix.

        In sampler.py line 75, the fix uses: `if tf > execute_on:`
        to determine which timeframes are "higher" and need offset.
        """
        execute_on = TimeFrame(1, TimeFrameUnit.Minute)

        # These should be identified as "higher" and offset
        tf_5min = TimeFrame(5, TimeFrameUnit.Minute)
        tf_1hour = TimeFrame(1, TimeFrameUnit.Hour)
        tf_1day = TimeFrame(1, TimeFrameUnit.Day)

        assert tf_5min > execute_on  # Should be offset
        assert tf_1hour > execute_on  # Should be offset
        assert tf_1day > execute_on  # Should be offset

        # Execute_on itself should NOT be offset
        execute_on_2 = TimeFrame(1, TimeFrameUnit.Minute)
        assert not (execute_on_2 > execute_on)  # Equal, no offset

        # Lower timeframes (if they existed) shouldn't be offset
        # (Though this scenario doesn't typically occur)
