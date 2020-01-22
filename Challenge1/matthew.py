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
import importlib
import os
import pickle
import string
import time
import cProfile
import pstats
from pstats import SortKey
from functools import partial
from itertools import permutations, islice
from math import factorial
from multiprocessing import Manager, TimeoutError
from multiprocessing.pool import Pool

# Function to validate the answer from Challenge1/main.py
validate_answer = importlib.import_module("main").validate_answer

# Timeout in seconds (seconds)
TIMEOUT = 60 * 60
START_TIME = int()
tree = dict()


def matthew(nums, multi=True):
    """
    Matthew's main function. Wrapper for the actual algorithm function.
    :param nums: (list of int) Numbers to form into a superpermutation
    :return: (string) Short superpermutation
    """
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
    # Initial superpermutation
    n = len(sp)
    tofind = set(allperms)  # Remaining permutations
    if sp in tofind:
        tofind.remove(sp)
    while tofind:
        for skip in range(1, n):
            # for all substrings of length "skip" in sp from "n" before
            # end to "skip" after that
            # EX: "abcdefg"[-4:][:3] would give "def"
            trial_add = None
            for trial_add in (''.join(p) for p in
                              permutations(sp[-n:][:skip])):
                # Append characters to end of "sp" and check if the last <n>
                # characters of the trial perm are a new permutation
                trial_perm = (sp + trial_add)[-n:]
                if trial_perm in tofind:
                    # Last "n" chars of "trial_perm" form a missing
                    # permutation, so we want to add it
                    sp += trial_add
                    # Remove found permutation from "tofind"
                    tofind.discard(trial_perm)

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
    return factorial(len(vals))


def depth(nums, multi=False):
    n = len(nums)
    chars = string.ascii_letters
    chars += string.digits

    multi = (n > 4) and multi

    # Too many to map,
    if n > len(chars):
        raise IOError(
            "ERROR: Input contains more than {} items.".format(len(chars)))

    # Map data to letters (a-z, A-Z, 0-9)
    mapping = dict()
    for idx, val in enumerate(nums):
        mapping[chars[idx]] = val
    nums = list(mapping.keys())

    # Run depth-first search
    perms = list(map(''.join, permutations(nums)))
    root = perms.pop(0)
    tofind = set(perms)

    global tree, START_TIME
    START_TIME = time.time()
    for i in range(1, len(nums) + 1):
        for c in permutations(nums, i):
            base = ''.join(list(c))
            remaining = ''.join([v for v in nums if v not in base])
            tree[base] = PermutationTree(base, remaining)

    allperms = map(''.join, permutations(nums))
    start = list(islice(allperms, 0, 1))
    best = greedy_construct(allperms, start[0])

    r = depth_first(root, n, tofind, best, top=multi)
    r = r if r is not None else best

    # Re-convert
    best = ""
    for char in r:
        best += str(mapping[char])

    # Return
    return best


def depth_first(root, n, tofind, best, top=False, master=None, lock=None):
    # If tofind is empty
    if not tofind:
        if master is not None:
            with lock:
                if len(root) < len(master.value):
                    master.value = root
                else:
                    root = master.value
        return root
    # If longer than the current best (assuming all unfound patterns can be
    # included with just one additional character each, which is the best case)
    elif (len(root) + len(tofind)) >= len(best):
        return None

    if time.time() > (START_TIME + TIMEOUT):
        return None

    # Generate potential branches based on the current root
    potential = try_add(root, n, tofind, best)
    if len(potential) == 0:
        return None

    # Potential branches collected, explore each one
    new = best
    # If more than one branch and has not previously done so,
    # do multiprocessing
    if top and (len(potential) > 1):
        for p in potential:
            p.n = n
            p.best = best

        args = [pickle.dumps(p) for p in potential]

        # Manager object to sync current best between processes
        with Manager() as manager:
            # Value to store the current best
            master = manager.Value(str, best)
            # Lock to synchronize access
            lock = manager.RLock()
            # Bind master and lock to function
            func = partial(depth_wrapper, master=master, lock=lock)
            # Create pool of workers
            pool = Pool(processes=min(len(args), os.cpu_count()), initializer=pool_init, initargs=(tree,))
            # Map potentials to bound functions run by pool
            async_result = pool.map_async(func, args)

            # Get async results, if timeout occurs, gets current shortest
            try:
                pool.close()  # No more tasks will be added to the pool
                # async_result.get(timeout=TIMEOUT)
                async_result.get()
            except TimeoutError:
                # Lock access to master and immediately terminate workers
                with lock:
                    pool.terminate()
                    # results = [master.value]
            # Clean up pool
            pool.join()

            new = master.value

        # Find shortest
        # new = min(results, key=len)

    else:
        # Don't spawn more processes, iterate over potential
        for p in potential:
            # Recursion
            r = depth_first(p.root, n, p.tofind, new, top=top, master=master, lock=lock)

            # If None, branch is discarded as not being a better solution
            if r is None:
                continue
            # Else, compare result to current best
            else:
                # Update current best from master
                if master is not None:
                    new = min(master.value, new, key=len)
                # print("# Branches:  {}".format(len(potential)))
                # print("Seed Level:   {}".format(
                #     npermutations(list(range(n))) - len(tofind)))
                # print("Current Best: {}".format(len(new)))

                # If current result is shorter than current best
                if len(r) < len(new):
                    new = r
                    # print("New Best:    {}\n".format(len(new)))

    # If a better solution was found, return it
    return new if len(new) < len(best) else None


def iterative_depth(base, master, lock):
    queue = [base]
    best = base.best
    evaluated = 0
    interval = 250
    while True:
        try:
            perm = queue.pop()
            root = perm.root
            tofind = perm.tofind
            n = perm.n
        except IndexError:
            break
        else:
            evaluated += 1
            if (evaluated % interval) == 0:
                best = min(best, master.value, key=len)
            # if (evaluated % (500*interval))==0:
            #     print("Eval: {}, Queue: {}".format(evaluated, len(queue)))
            # If tofind is empty
            if not tofind:
                with lock:
                    if len(root) < len(master.value):
                        master.value = root
                    else:
                        root = master.value
                best = root
                # print("Current Best: {}\n".format(len(best)))
                continue
            # If longer than the current best (assuming all unfound patterns can be
            # included with just one additional character each, which is the
            # best case)
            elif (len(root) + len(tofind)) >= len(best):
                # print("Too long, discarding. Queue Size:", len(queue))
                continue

            if time.time() > (START_TIME + TIMEOUT):
                break

            # Generate potential branches based on the current root
            potential = try_add(root, n, tofind, best)
            p_len = len(potential)
            if p_len == 0:
                continue
            else:
                for i in range(1, p_len+1):
                    p = potential[-i]
                    queue.append(p)
    return best


def try_add(root, n, tofind, best):
    potential = []
    for skip in range(1, n):
        trial_root = root[-(n-skip):]
        branch = tree[trial_root]
        perms_list = [i for i in branch.get_permutations() if i in tofind]
        for trial_perm in perms_list:
            trial_add = trial_perm.replace(trial_root, "")
            tmp = root + trial_add
            tmp_find = type(tofind)(tofind)
            tmp_find.discard(trial_perm)
            newperm = Permutation(root=tmp, n=n, tofind=tmp_find, add=trial_add)
            potential.append(newperm)

        # Added some branches, don't look for longer additions
        if len(potential) > 0:
            break
    return potential


def depth_wrapper(perm, master=None, lock=None):
    p = pickle.loads(perm)
    # cProfile.runctx(
    #     'iterative_depth(p, master=master, lock=lock)',
    #     globals(), locals(),
    #     os.path.join(os.getcwd(), "prof{}.prof".format(p.trial_add)))
    # depth_first(p.root, p.n, p.tofind, p.best, master=master, lock=lock)
    iterative_depth(p, master=master, lock=lock)


def pool_init(t):
    global tree, START_TIME
    tree = t
    START_TIME = time.time()


class Permutation(object):
    def __init__(self, root, tofind, n, best=None, add=None):
        super().__init__()
        self.root = root
        self.tofind = tofind
        self.n = n
        self.best = best
        self.trial_add = add


class PermutationTree:
    def __init__(self, base, remaining):
        self.base = base
        self.remaining = remaining
        self.leaves = None
        self.permutations = None
        self.calculate_children()

    def calculate_children(self):
        self.leaves = [''.join(p) for p in permutations(self.remaining)]
        self.permutations = self.calculate_permutations()

    def calculate_permutations(self):
        if (self.leaves is None) or (len(self.leaves) == 0):
            return [self.base]
        else:
            return [''.join([self.base, l]) for l in self.leaves]

    def get_permutations(self):
        return self.permutations

    def __str__(self):
        return "{} {}".format(self.base, self.leaves)


if __name__ == "__main__":
    N = 5

    # Test data
    # all_tests = [[1, 2], [1, 2, 21], [1, 2, 12], [1, 2, 3], [1, 2, 3, 4],
    #              [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6],
    #              [0, 1, 2, 3, 10, 11, 12, 13]]
    all_tests = [[j for j in range(1, i + 1)] for i in range(2, N + 1)]
    # all_tests = [[i for i in range(1, N + 1)]]

    for subdir, dirs, files in os.walk(os.getcwd()):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".prof":
                os.remove(os.path.join(subdir, file))

    for data in all_tests:
        print()
        print("Data: {}".format(data))

        # Run function
        start_time = time.time()
        super_permutation = matthew(data)
        end_time = time.time()

        # Verify
        # print(super_permutation)
        print("\rVerifying")
        validate_answer(data, super_permutation)

        print("time:  {:f} seconds".format(end_time - start_time))
        print("length:", len(super_permutation))

        for subdir, dirs, files in os.walk(os.getcwd()):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext == ".prof":
                    p = pstats.Stats(os.path.join(subdir, file))
                    p.sort_stats(SortKey.CUMULATIVE)
                    p.print_stats()
                    print()
                    p.print_callers()
