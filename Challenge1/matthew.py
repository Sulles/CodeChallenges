"""
Created on: 20 Dec, 2019

@author: Matthew Kesselring

Proven shortest:
N       L(N)
2   ==  3
3   ==  9
4   ==  33
5   ==  153
6   <=  872
7   <=  5907

"""
import cProfile
import logging
import operator
import pstats
import string
import time
import importlib
import pickle
from multiprocessing.pool import Pool
from collections import Counter
from functools import reduce
from itertools import permutations, islice
from math import factorial
from pstats import SortKey

validate_answer = importlib.import_module("main").validate_answer

# logger = logging.getLogger(__file__)
LOG_LEVEL = logging.WARNING


def matthew(nums, multi=False):
    """
    Matthew's main function. Wrapper for the actual algorithm function.
    :param nums: (list of int) Numbers to form into a superpermutation
    :return: (string) Short superpermutation
    """
    # Set up logger
    # logging.basicConfig(level=LOG_LEVEL)

    # Finds shortest: Depth first
    best = depth(nums, multi)

    # Return
    return best


def greedy_construct(allperms, start=None):
    """

    :param allperms:
    :return:
    """
    if start is not None:
        sp = start
    else:
        sp = list(islice(allperms, 0, 1))[0]
    # sp = list(islice(allperms, 0, 1))[0] if start is not None else start
    # Initial superpermutation
    n = len(sp)
    tofind = set(allperms)  # Remaining permutations
    # print("tofind", tofind)
    # print("sp:", sp)
    if sp in tofind:
        tofind.remove(sp)
    n_tofind = len(tofind)
    # logger.debug("looping...")
    while tofind:
        for skip in range(1, n):
            # for all substrings of length "skip" in sp from "n" before
            # end to "skip" after that
            # EX: "abcdefg"[-4:][:3] would give "def"
            trial_add = None
            for trial_add in (''.join(p) for p in
                              permutations(sp[-n:][:skip])):
                # logger.debug("")
                # logger.debug("Inside: trial_add loop")
                # logger.debug("sp: {}".format(sp))
                # logger.debug("trial_add: {}".format(trial_add))

                # Append characters to end of "sp" and check if the last <n>
                # characters of the trial perm are a new permutation
                trial_perm = (sp + trial_add)[-n:]
                if trial_perm in tofind:
                    # logger.debug("Inside: trial_perm loop")
                    # logger.debug("trial_perm: {}".format(trial_perm))

                    # Last "n" chars of "trial_perm" form a missing
                    # permutation, so we want to add it
                    sp += trial_add
                    # Remove found permutation from "tofind"
                    tofind.discard(trial_perm)

                    # Print progress
                    # if (LOG_LEVEL <= logging.INFO) and (
                    #         (len(tofind) % 50) == 0):
                    #     print("\rFound: {:.03f}%: {} of {}".format(
                    #         100 - ((len(tofind) / n_tofind) * 100),
                    #         n_tofind - len(tofind), n_tofind),
                    #         end="",
                    #         flush=True)

                    # Clear the chars we just added from the temp variable
                    trial_add = None
                    # Stop looking for an addition of this length, increment
                    # "skip" and look for a new length
                    break  # Break out of "for trial_add ..." loop

            # trial_add is None, break out of "for skip ..." loop. Checks if
            # "tofind" is empty, if not, restarts "for skip ..." loop
            if trial_add is None:
                break
    return sp


def npermutations(vals):
    num = factorial(len(vals))
    mults = Counter(vals).values()
    den = reduce(operator.mul, (factorial(v) for v in mults), 1)
    return num // den


def depth(nums, multi=False):
    n = len(nums)
    chars = string.ascii_letters
    chars += string.digits

    # Too many to map,
    if n > len(chars):
        raise IOError(
            "ERROR: Input contains more than {} items.".format(len(chars)))

    # Map data to letters (a-z, A-Z, 0-9)
    # logger.debug("Mapping items...")
    mapping = dict()
    for idx, val in enumerate(nums):
        mapping[chars[idx]] = val
    nums = list(mapping.keys())

    # Run depth-first search
    perms = map(''.join, permutations(nums))
    root = list(islice(perms, 1))[0]
    tofind = set(perms)
    allperms = map(''.join, permutations(nums))
    start = list(islice(allperms, 0, 1))
    best = greedy_construct(allperms, start[0])
    # logger.info(" Starting Best: {}".format(len(best)))
    r = depth_first(root, n, tofind, best, top=multi)
    r = r if r is not None else best

    # Re-convert
    best = ""
    for char in r:
        best += str(mapping[char])

    # Return
    return best


def depth_first(root, n, tofind, best, top=False):
    # If tofind is empty
    if not tofind:
        return root
    # If longer than the current best (assuming all unfound patterns can be
    # included with just one additional character each, which is the best case)
    elif (len(root) + len(tofind)) >= len(best):
        return None

    # Generate potential branches based on the current root
    potential = try_add(root, n, tofind, best)

    # Potential branches collected, explore each one
    new = best
    # If more than one branch and has not previously done so,
    # do multiprocessing
    if top and (len(potential) > 1):
        # print("Multi", potential)
        for p in potential:
            p.n = n
            p.best = best

        args = [pickle.dumps(p) for p in potential]

        # Start processes and get results
        pool = Pool()
        async_result = pool.map_async(depth_wrapper, args)
        pool.close()
        pool.join()
        results = async_result.get()

        # Find shortest
        for r in results:
            if r is None:
                continue
            # Else, compare result to current best
            else:
                # logger.info(" # Branches:  {}".format(len(potential)))
                if len(r) < len(new):
                    new = r
                    # logger.info(" New Best:    {}\n".format(len(new)))
    else:
        for p in potential:
            r = depth_first(p.root, n, p.tofind, new, top=top)
            # If None, branch is discarded as not being a better solution
            if r is None:
                continue
            # Else, compare result to current best
            else:
                # print(" # Branches:  {}".format(len(potential)))
                # print(" Seed Level:  {}".format(len(p.root) - n))
                # logger.info(" # Branches:  {}".format(len(potential)))
                # logger.info(" Seed Level:  {}".format(len(p.root) - n))
                if len(r) < len(new):
                    new = r
                    # logger.info(" New Best:    {}\n".format(len(new)))
                    # print(" New Best:    {}\n".format(len(new)))

    # If a better solution was found, return it
    return new if len(new) < len(best) else None


def try_add(root, n, tofind, best):
    potential = []
    for skip in range(1, n):
        # for all substrings of length "skip" in sp from "n" before
        # end to "skip" after that
        # EX: "abcdefg"[-4:][:3] would give "def"
        for trial_add in (''.join(p) for p in
                          permutations(root[-n:][:skip])):

            # Append characters to end of "sp" and check if the last <n>
            # characters of the trial perm are a new permutation
            trial_perm = (root + trial_add)[-n:]
            # print(root, trial_add, trial_perm)
            if trial_perm in tofind:
                # Last "n" chars of "trial_perm" form a missing
                # permutation, so we want to add it
                tmp = root + trial_add
                if (len(tmp) + len(tofind) - 1) < len(best):
                    # Remove found permutation from "tofind"
                    tmp_find = type(tofind)(tofind)
                    tmp_find.discard(trial_perm)
                    newperm = Permutation(tmp, tmp_find)
                    potential.append(newperm)

        # Added some branches, don't look for longer additions
        if len(potential) > 0:
            break
    return potential


def depth_wrapper(perm):
    p = pickle.loads(perm)
    return depth_first(p.root, p.n, p.tofind, p.best)


class Permutation(object):
    def __init__(self, root, tofind, n=None, best=None):
        super().__init__()
        self.root = root
        self.tofind = tofind
        self.n = n
        self.best = best


if __name__ == "__main__":
    # LOG_LEVEL = logging.INFO
    PROFILE = bool(1)
    MULTI = bool(1)
    N = 5

    # Test data
    # all_tests = [[1, 2], [1, 2, 21], [1, 2, 12], [1, 2, 3], [1, 2, 3, 4],
    #              [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6],
    #              [0, 1, 2, 3, 10, 11, 12, 13]]
    # all_tests = [[j for j in range(1, i + 1)] for i in range(2, N + 1)]
    all_tests = [[i for i in range(1, N + 1)]]

    for data in all_tests:
        print()
        print("Data: {}".format(data))

        if PROFILE:
            # prof = cProfile.Profile()
            # prof.runcall(matthew, data)
            # prof.dump_stats("matthew_profile")
            # p = pstats.Stats("matthew_profile")
            # p.sort_stats(SortKey.CUMULATIVE)
            # p.print_stats()
            # # print()
            # # p.print_callers()

            for m in [True, False]:
                times = []
                for i in range(5):
                    start_time = time.perf_counter()
                    sp = matthew(data, m)
                    end_time = time.perf_counter()
                    times.append(end_time-start_time)
                print("Multiprocessing: {:5s}, Average: {}".format(str(m), sum(times)/len(times)))
        else:
            # Run function
            start_time = time.perf_counter()
            super_permutation = matthew(data, MULTI)
            end_time = time.perf_counter()

            # Verify
            print("\rVerifying")
            validate_answer(data, super_permutation)

            print("time:  {:f} seconds".format(end_time - start_time))
            print("length:", len(super_permutation))
            # print("super permutation:", super_permutation)
