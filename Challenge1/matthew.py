"""
Created on: 20 Dec, 2019

@author: Matthew Kesselring

Proven shortest:
N       L(N)
2   ==  3
3   ==  9
4   ==  33
5   ==  153
6   <=  873
7   <=  5907
8   <=  48785

"""
import importlib
import os
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
from collections import deque

# Function to validate the answer from Challenge1/main.py
validate_answer = importlib.import_module("main").validate_answer

# Timeout in seconds
TIMEOUT = 60 * 0.5

START_TIME = float()
tree = dict()

shortest_known = {
    # 2: 3,
    # 3: 9,
    # 4: 33,
    # 5: 153,
    6: 873,
    7: 5913,
    8: 48785,
}


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

    multi = (n > 5) and multi

    # Too many to map,
    if n > len(chars):
        raise IOError(
            f"ERROR: Input contains more than {len(chars)} items.")

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


def depth_first(root, n, tofind, best, top=False):
    # Calculate number of processes
    if top:
        # num_processes = os.cpu_count() - 1
        num_processes = os.cpu_count() - 2
        # num_processes = 3 * os.cpu_count() // 4
        # num_processes = os.cpu_count() // 2
    else:
        num_processes = 1
    num_processes = max(num_processes, 1)

    # Generate potential branches based on the current root
    potential = try_add(root, n, tofind, best)
    if len(potential) == 0:
        return None
    while len(potential) < num_processes:
        p1 = potential.pop(0)
        tmp = try_add(p1.root, p1.n, p1.tofind, best)
        potential.extend(tmp)
    for p in potential:
        p.n = n
        p.best = best
    args = potential

    # Spawn processes if needed
    if num_processes == 1:
        solutions = []
        for a in args:
            solutions.append(iterative_depth(a, master=None, lock=None))
        new = min(solutions, key=len)
    else:
        # Manager object to sync current best between processes
        with Manager() as manager:
            # Value to store the current best
            master = manager.Value(str, best)
            # Lock to synchronize access
            lock = manager.RLock()
            # Bind master and lock to function
            func = partial(depth_wrapper, master=master, lock=lock)
            # Create pool of workers
            pool = Pool(processes=min(len(args), num_processes),
                        initializer=pool_init, initargs=(tree,))
            # Map potentials to bound functions run by pool
            async_result = pool.map_async(func, args)

            # Get async results, if timeout occurs, gets current shortest
            pool.close()  # No more tasks will be added to the pool
            async_result.get()
            # Clean up pool
            pool.join()

            new = master.value

    # If a better solution was found, return it
    return new if len(new) < len(best) else None


def iterative_depth(base, master, lock):
    stack = deque([base])
    best = base.best
    evaluated = 0
    count = 0
    interval = 500
    big_interval = 1000 * interval
    cached_shortest_known = shortest_known.get(base.n, 0)

    # Loop until break or timeout is reached
    END_TIME = START_TIME + TIMEOUT
    while time.time() < END_TIME:
        try:
            # Get item from top of stack and unpack
            perm = stack.pop()
            root = perm.root
            tofind = perm.tofind
            n = perm.n
        except IndexError:
            # Stack is empty
            break
        else:
            # Every 250 nodes, update cached best value
            evaluated += 1
            if (evaluated % interval) == 0:
                if master is not None:
                    best = min(best, master.value, key=len)
                if len(best) <= cached_shortest_known:
                    # An acceptably short solution has been found
                    break
                evaluated %= big_interval
                # count += 1
                # if evaluated == 0:
                #     print(f"Interval: {count}, Stack: {len(stack)}")

            # If tofind is empty, all permutations have been found
            if len(tofind) == 0:
                # Compare length with current best and master
                if master is not None:
                    with lock:
                        if len(root) < len(master.value):
                            master.value = root
                        else:
                            root = master.value
                    best = root
                else:
                    best = min(best, root, key=len)
                # print(f"Current Best: {len(best)}\n")
                continue
            # If longer than the current best (assuming all unfound patterns
            # can be included with just one additional character each,
            # which is the best case)
            elif (len(root) + len(tofind)) >= len(best):
                # Result from this node is too long, discard
                continue
            else:
                # Generate potential branches based on the current root and add
                # them to the stack
                potential = try_add(root, n, tofind, best)
                upper_lim = len(potential) + 1
                for i in range(1, upper_lim):
                    p = potential[-i]
                    stack.append(p)

    return best


def try_add(root, n, tofind, best):
    potential = []
    best_len = len(best)
    base_len = len(root) + len(tofind) - 1
    comp_len = best_len - base_len
    for skip in range(1, n):
        if skip >= comp_len:
            break
        trial_root = root[-(n-skip):]
        branch = tree[trial_root]
        perms_list = [i for i in branch.get_permutations() if i in tofind]
        for trial_perm in perms_list:
            trial_add = trial_perm.replace(trial_root, "")
            tmp = f"{root}{trial_add}"
            tmp_find = set(tofind)
            tmp_find.discard(trial_perm)
            newperm = Permutation(root=tmp, n=n, tofind=tmp_find, add=trial_add, best=best)
            potential.append(newperm)

        # Added some branches, don't look for longer additions
        if len(potential) > 0:
            break
    return potential


def depth_wrapper(perm, master=None, lock=None):
    # cProfile.runctx(
    #     'iterative_depth(p, master=master, lock=lock)',
    #     globals(), locals(),
    #     os.path.join(os.getcwd(), "prof{}.prof".format(p.trial_add)))
    iterative_depth(perm, master=master, lock=lock)


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
        return f"{self.base} {self.leaves}"


if __name__ == "__main__":
    N = 6

    # Test data
    # all_tests = [[1, 2], [1, 2, 21], [1, 2, 12], [1, 2, 3], [1, 2, 3, 4],
    #              [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6],
    #              [0, 1, 2, 3, 10, 11, 12, 13]]
    # all_tests = [[j for j in range(1, i + 1)] for i in range(2, N + 1)]
    all_tests = [[i for i in range(1, N + 1)]]

    for subdir, dirs, files in os.walk(os.getcwd()):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".prof":
                os.remove(os.path.join(subdir, file))

    for data in all_tests:
        print()
        print(f"Data: {data}")

        # Run function
        start_time = time.time()
        super_permutation = matthew(data)
        end_time = time.time()

        # Verify
        # print(super_permutation)
        print("\rVerifying")
        validate_answer(data, super_permutation)

        print(f"time:  {(end_time-start_time):f} seconds")
        print(f"length: {len(super_permutation)}")

        for subdir, dirs, files in os.walk(os.getcwd()):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext == ".prof":
                    p = pstats.Stats(os.path.join(subdir, file))
                    p.sort_stats(SortKey.CUMULATIVE)
                    p.print_stats()
                    print()
                    p.print_callers()
