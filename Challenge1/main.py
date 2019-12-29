"""
=== CHALLENGE #1 ===

--- Description ---
From a list of integers, find the shortest string which contains all permutations of those numbers.
For example: [1, 2] has "121" as the shortest string which contains all permutations of 1, 2, and 21.

--- Allowable imports ---
numpy
itertools
"""

from itertools import permutations
from os import walk, getcwd
from time import time

ignore = ['main.py', '__init__.py', 'profiler.py', 'log_output.txt']

all_tests = [[1, 2, 3, 4]]
# [[1, 2], [1, 2, 21], [1, 2, 12], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
# [1, 2, 3, 4, 5, 6]
# [0, 1, 2, 3, 10, 11, 12, 13]


def main():
    """
    Main function will:
     - Get all file names
     - Try and import function names that match file names in this folder
     - Start timer when sending a list of integers to the function
     - End timer when a response is received
     - Validate that the string has all possible permutations
     - Print results
    TODO: Update a ranking excel sheet with the results
    """
    # Get all files in path
    all_files = list()
    for (dir_path, _, file_names) in walk(getcwd()):
        all_files.append(file_names)
    all_files = all_files[0]  # only want .py files

    # print('all files: {}'.format(all_files))

    # import all functions
    all_methods = list()
    for file in all_files:
        if not in ignore:
            print('Got valid file: {}'.format(file))
            try:
                method = import_from(file[0:-3], file[0:-3])
                all_methods.append(method)
            except Exception as e:
                print('ERROR: {}'.format(e))
                print('Failed to get method: {} from file {}'.format(file[0:-3], file))
                pass

    for method in all_methods:
        for test in all_tests:
            print('\nSending {} to {}'.format(test, method))
            start_time = time()
            answer = method(test)
            elapsed_time = time() - start_time
            print('Method returned string of length {} and took {} seconds'.format(len(answer), elapsed_time))
            validate_answer(test, answer)


def validate_answer(test_set, answer):
    """ This method will verify that all permutations of a set of numbers exists in the answer """
    if type(answer) is not str:
        print('FAIL: Answer type is not string, was provided {}'.format(type(answer)))
        return

    all_permutations = list(permutations(test_set))
    individual_perms = list()
    for perm in all_permutations:
        individual_perms.append(''.join([str(_) for _ in perm]))

    for perm in individual_perms:
        if perm not in answer:
            print('FAIL: Answer did not contain {}!'.format(perm))
            return

    print('SUCCESS!!!')


def import_from(module, name):
    """ This method tries to import all methods from challenger files in this folder """
    try:
        module = __import__(module, fromlist=[name])
        return getattr(module, name)
    except Exception as e:
        print('ERROR: {}'.format(e))
        print('Failed to load {}, skipping'.format(name))
        raise


if __name__ == "__main__":
    main()
