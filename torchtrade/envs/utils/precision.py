"""How many decimals a venue's tick or lot step implies (#278)."""

from decimal import Decimal, InvalidOperation


def decimals_for_step(step) -> int:
    """Decimal places implied by `step`, e.g. 0.001 -> 3, 1e-06 -> 6, 1 -> 0.

    Every venue used to derive this by string-inspecting the step:

        len(str(step).rstrip('0').split('.')[-1]) if '.' in str(step) else 0

    which reads `str(1e-06)` as `'1e-06'`, finds no `'.'`, and answers 0. Rounding a
    quantity to 0 decimals turned 0.977 into 1.0 -- bybit's replay path then asked for a
    whole BTC against a balance that covered 0.977, and every open was refused with a
    `logger.warning` and no raise. The symptom is a replay that reports a flat strategy
    for a policy that is trading, which is exactly the failure #278 exists to remove.

    Decimal reads the exponent instead of the repr, so notation cannot change the answer.
    """
    try:
        # Integral steps normalize to a POSITIVE exponent (1000 -> 1E+3); they imply no
        # decimals, and max() keeps that from becoming a negative ndigits. int() is
        # INSIDE the try because NaN and Infinity are valid Decimals whose exponent is
        # the string 'n'/'F', so converting them raises where the guard promised 0.
        return max(0, -int(Decimal(str(step)).normalize().as_tuple().exponent))
    except (InvalidOperation, ValueError, TypeError):
        return 0
