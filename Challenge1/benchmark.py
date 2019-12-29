"""
This is the benchmark file for Challenge #1. It simply concatenates all permutations into one string.
"""

from itertools import permutations


def benchmark(input_list):
    all_permutations = list(permutations(input_list))
    all_sets = ''
    for perm in all_permutations:
        all_sets = all_sets + ''.join([str(_) for _ in perm])
    return all_sets
