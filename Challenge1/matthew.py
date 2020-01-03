"""
Created on: 20 Dec, 2019

@author: Matthew Kesselring
"""
from itertools import permutations
from functools import reduce
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from math import factorial
from collections import Counter
import operator
import re
import numpy as np
import logging
import time
import string
import random
from copy import deepcopy
from threading import Thread
from queue import Queue
from Challenge1.main import validate_answer

logger = None
LOG_LEVEL = logging.WARNING


def matthew(nums):
    """
    Matthew's main function. Wrapper for the actual algorithm function.
    :param nums: (list of int) Numbers to form into a superpermutation
    :return: (string) Short superpermutation
    """
    # Set up logger
    logging.basicConfig(level=LOG_LEVEL)
    global logger
    logger = logging.getLogger(__file__)

    # Initial attempt: Random sampling
    # best = samples(nums)

    # Better algorithm: Greedy appending
    best = greedy(nums)

    # Return
    return best


def greedy(nums):
    """
    Creates multiple threads to try different permutation orders
    :param nums:
    :return:
    """
    best = ""
    best_len = float('inf')
    # Try it a couple times, with shuffled data
    pool = ThreadPool()
    args = []
    for i in range(cpu_count()):
        args.append(deepcopy(nums))
    # Start async processes
    async_result = pool.map_async(greedy_worker, args)
    results = async_result.get()

    # n = len(nums)
    # chars = string.ascii_letters
    # chars += string.digits
    #
    # # Too many to map,
    # if n > len(chars):
    #     raise IOError(
    #         "ERROR: Input contains more than {} items.".format(len(chars)))
    #
    # # Map data to letters (a-z, A-Z, 0-9)
    # logger.debug("Mapping items...")
    # mapping = dict()
    # for idx, val in enumerate(nums):
    #     mapping[chars[idx]] = val
    # nums = mapping.keys()
    #
    # # Loop, adding chars until an unseen permutation is formed in the last "n"
    # # chars
    # logger.debug("Generating permutations...")
    # allperms = [''.join(p) for p in permutations(nums)]
    #
    # # Try out different permutations
    # pool = ThreadPool()
    # args = []
    # logger.info("len allperms: {}".format(len(allperms)))
    # for i in range(len(allperms)):
    #     tmp = deepcopy(allperms)
    #     tmp.insert(0, tmp.pop(i))
    #     args.append(tmp)
    # async_result = pool.map_async(greedy_construct, args)
    # results = async_result.get()
    #
    # random.shuffle(allperms)
    # sp = greedy_construct(allperms)
    #
    # # Re-convert to given data
    # logger.debug("Re-converting")
    # best = ""
    # for char in sp:
    #     best += str(mapping[char])

    pool.close()
    pool.join()

    for r in results:
        r_len = len(r)
        # Compare current result to best
        if (best_len == float('inf')) or (r_len < best_len):
            best = r
            best_len = r_len
            logger.info("New best length: {}".format(best_len))
            logger.info("New best permutation: {}".format(best))

    # Return
    return best


def greedy_worker(nums):
    """
    Maps each element of the source data to a unique character and calls the
    construction function to generate the superpermutation
    :param nums: (list of int) Numbers to form into a superpermutation
    :return: (string) Short superpermutation
    """
    n = len(nums)
    chars = string.ascii_letters
    chars += string.digits

    # Too many to map,
    if n > len(chars):
        raise IOError(
            "ERROR: Input contains more than {} items.".format(len(chars)))

    # Map data to letters (a-z, A-Z, 0-9)
    logger.debug("Mapping items...")
    mapping = dict()
    for idx, val in enumerate(nums):
        mapping[chars[idx]] = val
    nums = mapping.keys()

    # Loop, adding chars until an unseen permutation is formed in the last "n"
    # chars
    logger.debug("Generating permutations...")
    allperms = [''.join(p) for p in permutations(nums)]

    # Try out different permutations
    random.shuffle(allperms)
    sp = greedy_construct(allperms)

    # Re-convert to given data
    logger.debug("Re-converting")
    best = ""
    for char in sp:
        best += str(mapping[char])
    return best


def greedy_construct(allperms):
    """

    :param allperms:
    :return:
    """
    sp = allperms[0]  # Initial superpermutation
    n = len(sp)
    tofind = set(allperms[1:])  # Remaining permutations
    n_tofind = len(tofind)
    logger.debug("looping...")
    while tofind:
        for skip in range(1, n):
            # for all substrings of length "skip" in sp from "n" before
            # end to "skip" after that
            # EX: "abcdefg"[-4:][:3] would give "def"
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


def samples(nums, ratio=int(1e5)):
    """
    Generates all possible superpermutations and then collapses a random sample
    :param nums: (list of int) Numbers to form into a superpermutation
    :param ratio: (int) Optional number of random samples
    :return: (string) Short superpermutation
    """
    # Get list of all permutations
    p_list = get_permutations(nums)

    # Random sampling
    logger.info("Calculating number of samples")
    num_combos = npermutations(p_list)
    n_sample = ratio if (ratio < num_combos) else num_combos

    # Set up loop
    logger.info("Setting up loop")
    best = ""
    best_len = float("inf")
    chunk = []
    n = 0
    p = ThreadPool()
    logger.info("Looping")
    for m in range(1, int(n_sample + 1)):
        if n_sample == num_combos:
            all_combos = permutations(p_list)
            for count in range(num_combos):
                current = get_random_permutation(all_combos, random=False)
                chunk.append(current)
                n += 1
        else:
            current = get_random_permutation(p_list)
            chunk.append(current)
            n += 1

        if (len(chunk) >= 100) or (n == n_sample) or (n_sample == num_combos):
            # Collapse in threads
            async_result = p.map_async(iterate_thread, chunk)
            result = async_result.get()
            chunk = []

            percent = (n / n_sample) * 100
            print("\r{:.04f}%: {} of {}".format(percent, n, n_sample),
                  end="",
                  flush=True)

            # Get shortest
            tmp = min(result, key=len)
            if len(tmp) < best_len:
                best = tmp
                best_len = len(tmp)
                print()
                logger.debug("new best: {}".format(best))
                logger.debug("new length: {}".format(len(best)))

        if n_sample == num_combos:
            break

    # Clean up pool
    p.close()
    p.join()

    return best


def get_random_permutation(vals, random=True):
    if random:
        yield np.random.permutation(vals)
    else:
        yield next(vals)


def npermutations(vals):
    num = factorial(len(vals))
    mults = Counter(vals).values()
    den = reduce(operator.mul, (factorial(v) for v in mults), 1)
    return num // den


def iterate_thread(current):
    current = list(next(current))
    # Join into super-permutation
    super_p = ",".join(list(current))
    super_p = ",{},".format(super_p)

    # Collapse
    collapsed = collapse(super_p, current)

    return collapsed


def get_permutations(nums):
    # Convert to strings
    for idx, n in enumerate(nums):
        nums[idx] = str(n)

    # Get permutations
    p_list = list(permutations(nums))

    # Convert each permutation from tuple to string
    for idx, p in enumerate(p_list):
        p_list[idx] = ",".join(list(p))
    return p_list


def collapse(super_p, p_list):
    # Remove neighboring duplicates
    logger.debug("Before: {}".format(super_p))
    collapsed = re.sub(r'(,)([0-9]+),(?=\1)', r'', super_p)
    logger.debug("After:    {}".format(collapsed))
    # Remove commas used for delimiting
    collapsed = collapsed.replace(",", "")
    return collapsed


if __name__ == "__main__":
    LOG_LEVEL = logging.INFO

    # Test data
    all_tests = [[1, 2], [1, 2, 21], [1, 2, 12], [1, 2, 3], [1, 2, 3, 4],
                 [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6],
                 [0, 1, 2, 3, 10, 11, 12, 13]]
    # all_tests = [[i for i in range(1, 5)]]

    for data in all_tests:
        print()
        print("Data: {}".format(data))
        # Run function
        super_permutation = matthew(data)

        # Verify
        print("Verifying")
        validate_answer(data, super_permutation)
        # for p in get_permutations(data):
        #     n = p.replace(",", "")
        #     if p not in super_permutation:
        #         print("ERROR: {} not in {}".format(n, super_permutation))
        #         break
        # else:
        #     print("SUCCESS")

        print()
        print("data:", data)
        print("length:", len(super_permutation))
        print("super permutation:", super_permutation)

        time.sleep(0.5)
