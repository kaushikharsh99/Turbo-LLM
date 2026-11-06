import math


def validate_generation(max_new_tokens, temperature, top_p):
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
