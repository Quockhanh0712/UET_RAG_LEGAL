"""Provision/article window splitting helpers."""

from __future__ import annotations

from typing import Callable, List, Sequence


def build_unit_windows(
    units: Sequence[object],
    *,
    token_getter: Callable[[object], int],
    target_chunk_tokens: int,
    hard_cap_tokens: int,
    overlap_tokens: int,
) -> List[List[object]]:
    if not units:
        return []

    windows: List[List[object]] = []
    cursor = 0
    total_units = len(units)
    while cursor < total_units:
        current_units: List[object] = []
        token_total = 0
        idx = cursor

        while idx < total_units:
            unit = units[idx]
            unit_tokens = int(token_getter(unit))
            if current_units and token_total + unit_tokens > target_chunk_tokens:
                break
            if current_units and token_total + unit_tokens > hard_cap_tokens:
                break
            current_units.append(unit)
            token_total += unit_tokens
            idx += 1
            if token_total >= target_chunk_tokens:
                break

        if not current_units:
            current_units = [units[cursor]]
            idx = cursor + 1

        windows.append(current_units)

        if idx >= total_units:
            break

        next_cursor = idx
        overlap_sum = 0
        back = idx - 1
        while back > cursor and overlap_sum < overlap_tokens:
            overlap_sum += int(token_getter(units[back]))
            next_cursor = back
            back -= 1
        cursor = next_cursor if next_cursor > cursor else min(idx, cursor + 1)

    return windows

