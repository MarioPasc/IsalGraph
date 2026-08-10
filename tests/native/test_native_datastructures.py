"""Differential tests for the ported data structures.

Both the CDLL free list and the displacement-pair ordering are load-bearing:
a deviation in either silently changes emitted strings rather than raising.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("isalgraph.core._native", reason="C++ extension not built")

from isalgraph.core import _native as ext
from isalgraph.core.cdll import CircularDoublyLinkedList
from isalgraph.core.graph_to_string import generate_pairs_sorted_by_sum
from isalgraph.errors import CapacityError

# ----------------------------------------------------------------------
# CDLL
# ----------------------------------------------------------------------


def test_free_list_allocates_zero_one_two_in_order() -> None:
    """`list(range(capacity-1, -1, -1)).pop()` yields 0, then 1, then 2..."""
    py = CircularDoublyLinkedList(8)
    cpp = ext.Cdll(8)
    prev_py = py.insert_after(-1, 100)
    prev_cpp = cpp.insert_after(-1, 100)
    assert prev_py == prev_cpp == 0
    for k in range(1, 8):
        a = py.insert_after(prev_py, 100 + k)
        b = cpp.insert_after(prev_cpp, 100 + k)
        assert a == b == k, (k, a, b)
        prev_py, prev_cpp = a, b


def test_removed_index_is_reused_lifo() -> None:
    """remove() pushes onto the free stack, so the next insert reclaims it."""
    py = CircularDoublyLinkedList(6)
    cpp = ext.Cdll(6)
    nodes_py = [py.insert_after(-1 if i == 0 else i - 1, i * 10) for i in range(4)]
    nodes_cpp = [cpp.insert_after(-1 if i == 0 else i - 1, i * 10) for i in range(4)]
    assert nodes_py == nodes_cpp

    py.remove(2)
    cpp.remove(2)
    assert py.insert_after(0, 99) == cpp.insert_after(0, 99) == 2


@pytest.mark.parametrize("seed", range(40))
def test_random_operation_sequences_agree(seed: int) -> None:
    """Drive both CDLLs through identical random op streams and diff the state."""
    rng = random.Random(seed)
    capacity = rng.randint(3, 12)
    py = CircularDoublyLinkedList(capacity)
    cpp = ext.Cdll(capacity)

    live: list[int] = []
    first_py = py.insert_after(-1, 0)
    first_cpp = cpp.insert_after(-1, 0)
    assert first_py == first_cpp
    live.append(first_py)

    for step in range(60):
        if live and (len(live) >= capacity or rng.random() < 0.35):
            victim = live[rng.randrange(len(live))]
            py.remove(victim)
            cpp.remove(victim)
            live.remove(victim)
            if not live:
                first_py = py.insert_after(-1, step)
                first_cpp = cpp.insert_after(-1, step)
                assert first_py == first_cpp
                live.append(first_py)
        else:
            anchor = live[rng.randrange(len(live))]
            a = py.insert_after(anchor, step)
            b = cpp.insert_after(anchor, step)
            assert a == b, (seed, step, a, b)
            live.append(a)

        assert py.size() == cpp.size()
        for node in live:
            assert py.get_value(node) == cpp.get_value(node)
            assert py.next_node(node) == cpp.next_node(node)
            assert py.prev_node(node) == cpp.prev_node(node)


def test_capacity_exhaustion_message_matches_reference() -> None:
    py = CircularDoublyLinkedList(2)
    cpp = ext.Cdll(2)
    py.insert_after(-1, 0)
    py.insert_after(0, 1)
    cpp.insert_after(-1, 0)
    cpp.insert_after(0, 1)

    with pytest.raises(RuntimeError) as py_exc:
        py.insert_after(0, 2)
    with pytest.raises(CapacityError) as cpp_exc:
        cpp.insert_after(0, 2)
    assert str(py_exc.value) == str(cpp_exc.value) == "CircularDoublyLinkedList is full"
    # The builtin-mixin half of this contract is asserted in
    # test_native_errors.py, gated on errors.py landing at integration.


@pytest.mark.parametrize("steps", [-7, -3, -1, 0, 1, 4, 11])
def test_walk_matches_repeated_next_prev(steps: int) -> None:
    cpp = ext.Cdll(5)
    prev = cpp.insert_after(-1, 0)
    for i in range(1, 5):
        prev = cpp.insert_after(prev, i)

    expected = 0
    for _ in range(abs(steps)):
        expected = cpp.next_node(expected) if steps > 0 else cpp.prev_node(expected)
    assert cpp.walk(0, steps) == expected


# ----------------------------------------------------------------------
# Displacement pairs
# ----------------------------------------------------------------------


@pytest.mark.parametrize("m", list(range(1, 25)))
def test_pair_ordering_is_byte_identical(m: int) -> None:
    """Sort key must be (|a|+|b|, |a|, (a, b)) -- all three components."""
    expected = generate_pairs_sorted_by_sum(m)
    got = ext.pairs_sorted_by_cost(m)
    assert got == expected


def test_pair_ordering_is_not_the_algebraic_sum_bug() -> None:
    """Regression guard for historical bug B2 (sorting by a + b)."""
    pairs = ext.pairs_sorted_by_cost(3)
    costs = [abs(a) + abs(b) for a, b in pairs]
    assert costs == sorted(costs)
    assert pairs[0] == (0, 0)
    # (-1, 0) and (1, 0) both cost 1; the algebraic-sum bug would rank
    # (-1, -1) (sum -2) ahead of them.
    assert abs(pairs[1][0]) + abs(pairs[1][1]) == 1


def test_pairs_are_memoised() -> None:
    """The reference rebuilds this list at every recursion frame; we cache it."""
    before = ext.pairs_cache_size()
    first = ext.pairs_sorted_by_cost(9)
    mid = ext.pairs_cache_size()
    second = ext.pairs_sorted_by_cost(9)
    assert first == second
    assert ext.pairs_cache_size() == mid
    assert mid >= before


@pytest.mark.parametrize("m", [0, -1, -7])
def test_pairs_rejects_non_positive_m(m: int) -> None:
    with pytest.raises(ValueError) as ref_exc:
        generate_pairs_sorted_by_sum(m)
    with pytest.raises(ValueError) as cpp_exc:
        ext.pairs_sorted_by_cost(m)
    assert str(cpp_exc.value) == str(ref_exc.value) == "m must be a positive integer."
