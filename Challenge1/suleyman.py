"""
Created December 19, 2019
@author: SBARTHE6
"""

from copy import copy
from itertools import permutations


class Perm:
    def __init__(self, string, log):
        assert type(string) is str, 'Must provide string input'
        self.string = string
        self.log = log
        self.log.writelines('\n\nSuccessfully initialized Perm object: "{}"'.format(self.string))

        self.overlap_perm_results = list()

    def overlap(self, other_perm, best_overlap_len=None):
        """
        Find the overlap between one permutation object and another
        :param other_perm: Other permutation object that is to be compared against
        :param best_overlap_len: current highest overlap count
        :return: Returns multiple things:
            - String: The overlap found between the two
            - Int: Index of overlap in larger string
            - Perm: self
        """
        assert type(other_perm) is Perm, 'Permutation objects can only find overlap with other permutation objects!'
        assert len(other_perm.string) <= len(self.string), 'Only compare equal or smaller length strings!'

        self.log.writelines('\n-- OVERLAP --')

        other_string = copy(other_perm.string)

        if other_string in self.string:
            self.log.writelines('\n\n (*) CRITICAL ERROR: other permutation should not exist in self!\n')
            return None, None, other_perm, True

        total_overlapped = ''
        overlap_1 = None
        overlap_2 = None
        index_1 = None
        index_2 = None
        self.log.writelines('\nLooking for overlap between {} and {}'.format(self.string, other_string))
        for length in range(1, len(other_string)):
            possible_overlap_1 = other_string[length:]  # end of smaller string, beginning of base string
            possible_overlap_2 = other_string[:len(possible_overlap_1)]  # start of smaller string, end of base string

            if best_overlap_len is not None and len(possible_overlap_1) < best_overlap_len:
                self.log.writelines('\n1. Possible overlap 1 "{}" length was shorter than current best overlap length: {}'.format(
                    possible_overlap_1, best_overlap_len))
                if len(possible_overlap_1) != len(possible_overlap_2):
                    self.log.writelines('\n\n (*) CRITICAL ERROR: Possible overlaps should be the same length!\n')
                return None, None, None, False

            self.log.writelines('\n1. Checking {} in substring of longer string: {}'.format(
                possible_overlap_1, self.string[:len(possible_overlap_1)]))
            if possible_overlap_1 == self.string[:len(possible_overlap_1)] and \
                    possible_overlap_1 != "":
                index_1 = self.string.index(possible_overlap_1)
                self.log.writelines('\n1. Identified overlap "{}" at index {} in longer string {}'.format(
                    possible_overlap_1, index_1, self.string))
                if len(str(possible_overlap_1)) >= len(total_overlapped):
                    total_overlapped = possible_overlap_1
                    overlap_1 = possible_overlap_1

            self.log.writelines('\n2. Checking {} in substring of longer string: {}'.format(
                possible_overlap_2, self.string[len(self.string) - len(possible_overlap_2):]))
            if possible_overlap_2 == self.string[len(self.string) - len(possible_overlap_2):] and \
                    possible_overlap_2 != "":
                index_2 = self.string.rindex(possible_overlap_2)
                self.log.writelines('\n2. Identified overlap "{}" at index {} in longer string {}'.format(
                    possible_overlap_2, index_2, self.string))
                if len(str(possible_overlap_1)) >= len(total_overlapped):
                    total_overlapped = possible_overlap_2
                    overlap_2 = possible_overlap_2

            # Exit out early if overlap is found
            if overlap_1 is not None and overlap_2 is not None:
                return [overlap_1, overlap_2], [index_1, index_2], other_perm, False
            if overlap_1 is not None:
                return overlap_1, index_1, other_perm, False
            if overlap_2 is not None:
                return overlap_2, index_2, other_perm, False

        return None, None, None, False

    def shove_in(self, overlapped_stuff, index_at_self, next_string):
        self.log.writelines('\n-- SHOVE IN --')
        self.log.writelines('\nFrom current string "{}", trying to shove in"{}", which has overlap: "{}" and index of self:{}'.format(
            self.string, next_string, overlapped_stuff, index_at_self))

        if index_at_self != 0:
            self.log.writelines('\nOverlap between self "{}" and other string "{}" is at START of other string'.format(
                self.string, next_string))
            unique_next_string = next_string[len(overlapped_stuff):]
            self.log.writelines('\nUnique string: "{}"'.format(unique_next_string))

            self.string = self.string + unique_next_string
            self.log.writelines('\nSelf has new string: "{}"'.format(self.string))

        else:
            self.log.writelines('\nOverlap between self "{}" and other string "{}" is at END of other string'.format(
                self.string, next_string))
            unique_next_string = next_string[:-len(overlapped_stuff)]
            self.log.writelines('\nUnique string: "{}"'.format(unique_next_string))

            self.string = unique_next_string + self.string
            self.log.writelines('\nSelf has new string: "{}"'.format(self.string))


class Level:
    def __init__(self, root_perm, all_perms, log):
        assert type(all_perms) is list and [type(_) is Perm for _ in all_perms] and type(root_perm) is Perm
        self.root_perm = root_perm
        self.all_perms = all_perms
        self.log = log
        self.log.writelines('\nSuccessfully initialized a new level with {} perms'.format(len(self.all_perms)))

    def get_len_sub_perms(self):
        return copy(len(self.all_perms))

    def add_perm(self, perm):
        self.all_perms.append(perm)

    def get_next_optimal_levels(self):
        # Sanitize
        for perm in self.all_perms:
            if perm.string in self.root_perm.string:
                self.log.writelines('\n- Removed duplicate')
                self.all_perms.pop(self.all_perms.index(perm))

        # Check if any permutations are left, if not, return root as solution
        if len(self.all_perms) == 0:
            return [self.root_perm]

        self.log.writelines('\n\nTrying to find next optimal levels'.format(len(self.all_perms)))
        self.log.writelines('\nStarting with: "{}"'.format(self.root_perm.string))
        self.log.writelines('\nAll other perms: {}'.format([_.string for _ in self.all_perms]))
        all_overlaps = list()
        all_overlap_lengths = list()
        all_overlap_indexes = list()
        all_tested_perms = list()
        current_max_len = 0
        for x in range(len(self.all_perms)):
            overlap, index, perm, is_duplicate = self.root_perm.overlap(self.all_perms[x], current_max_len)
            self.log.writelines('\nGot overlap: {}'.format(overlap))
            self.log.writelines('\nGot index: {}'.format(index))
            self.log.writelines('\nGot perm: {}'.format(perm))
            if type(overlap) is list:
                [all_overlaps.append(_) for _ in overlap]
                [all_overlap_lengths.append(len(_)) for _ in overlap]
                [all_overlap_indexes.append(_) for _ in index]
                [all_tested_perms.append(perm) for _ in index]
            elif overlap is not None:
                all_overlaps.append(overlap)
                all_overlap_lengths.append(len(overlap))
                all_overlap_indexes.append(index)
                all_tested_perms.append(perm)

            if len(all_overlap_lengths) > 0 and max(all_overlap_lengths) > current_max_len:
                current_max_len = max(all_overlap_lengths)
                self.log.writelines('\nGot new maximum length overlap! Is length: {}'.format(current_max_len))

        self.log.writelines('\nGot all overlaps: "{}"'.format(all_overlaps))
        number_of_maximum_overlaps = all_overlap_lengths.count(current_max_len)
        self.log.writelines('\nThere are {} occurrences of maximum length'.format(number_of_maximum_overlaps))

        optimal_index = list()
        optimal_overlap = list()
        optimal_overlap_index = list()
        optimal_combination_perm = list()
        for x in range(number_of_maximum_overlaps):
            # get optimal overlaps
            optimal_index.append(all_overlap_lengths.index(current_max_len))
            optimal_overlap.append(all_overlaps[optimal_index[-1]])
            optimal_overlap_index.append(all_overlap_indexes[optimal_index[-1]])
            optimal_combination_perm.append(all_tested_perms[optimal_index[-1]])
            self.log.writelines('\nGot optimal overlap: "{}", has index: {} in all overlaps.\n'
                           'Base string: "{}" has overlap "{}" with "{}"'.
                format(optimal_overlap[-1], optimal_index[-1],
                       self.root_perm.string, all_overlaps[optimal_index[-1]], optimal_combination_perm[-1].string))
            # remove max_length from all overlap lengths
            all_overlap_lengths[optimal_index[-1]] = 0

        response = list()
        for x in range(len(optimal_overlap)):
            new_root_perm = copy(self.root_perm)
            new_root_perm.shove_in(optimal_overlap[x], optimal_overlap_index[x], optimal_combination_perm[x].string)
            new_other_perms = [_ for _ in self.all_perms if _ != optimal_combination_perm[x]]
            if len(new_other_perms) == 0:
                response.append(new_root_perm)
            else:
                self.log.writelines('\n({}) New perm: "{}"'.format(x, new_root_perm.string))
                self.log.writelines('\n({}) Other perms: {}'.format(
                    x, [_.string for _ in self.all_perms if _ != optimal_combination_perm[x]]))
                response.append(Level(
                    new_root_perm, [_ for _ in self.all_perms if _ != optimal_combination_perm[x]], self.log))
        return response


class Tree:
    def __init__(self, log):
        self.levels = list()
        self.index = 0
        self.solutions = list()
        self.log = log

    def add_new_level(self, level):
        self.levels.append(level)

    def generate_next_levels(self):
        self.log.writelines('\nWorking with level #{}'.format(len(self.levels)))
        response = [_ for _ in self.levels[-1].get_next_optimal_levels()]
        self.log.writelines('\nGot response length: {}'.format(len(response)))
        for r in response:
            if type(r) is Perm:
                self.log.writelines('\n-- Found solution -- "{}"\n'.format(r.string))
                self.solutions.append(r)
            elif type(r) is Level:
                self.log.writelines('\n-- Got new Level --')
                self.levels.append(r)


def suleyman(input_list):

    with open('log_output.txt', 'w') as LOG:
        
        TREE = Tree(LOG)

        all_perms = list()
        all_permutations = list(permutations(input_list))
        unique_permutations = list()
        # Get unique permutations
        for perm in all_permutations:
            if perm not in unique_permutations:
                unique_permutations.append(perm)
        # Generate all permutation objects
        for perm in unique_permutations:
            string = ''.join([str(_) for _ in perm])
            all_perms.append(Perm(string, LOG))
        # Create first level in tree
        TREE.add_new_level(Level(all_perms[0], all_perms[1:], LOG))

        iteration = 0
        iter_max = 1000
        while len(TREE.solutions) == 0:
            LOG.writelines('\nIteration {}'.format(iteration))
            TREE.generate_next_levels()

            if iteration == iter_max:
                LOG.writelines('\nIterations surpassed break-off point: {}'.format(iter_max))
                LOG.writelines('\nERROR??')
                exit()
            else:
                iteration += 1

        lengths_of_all_solutions = [len(_.string) for _ in TREE.solutions]
        len_of_shortest_solution = min(lengths_of_all_solutions)
        index_of_shortest_sol = lengths_of_all_solutions.index(len_of_shortest_solution)
        shortest_solution = TREE.solutions[index_of_shortest_sol]
        # LOG.writelines('\n\nANSWER: "{}"\n'.format(shortest_solution.string))

        # all_permutations = list(permutations(input_list))
        # all_string_perms = list()
        # for perm in all_permutations:
        #     string = ''.join([str(_) for _ in perm])
        #     all_string_perms.append(string)
        #
        # # LOG.writelines('\nAll string permutations: {}'.format(all_string_perms))
        #
        # copy_of_all_string_perms = copy(all_string_perms)
        # answer = copy_of_all_string_perms[0]
        # copy_of_all_string_perms.pop(0)     # remove first index
        # for string in all_string_perms[1:]:
        #     if string not in answer:
        #         overlap, index = get_overlap(answer, string)
        #         if overlap is not None:
        #             answer = shove_in(answer, overlap, index, string)
        #             copy_of_all_string_perms.pop(copy_of_all_string_perms.index(string))
        #     else:
        #         # LOG.writelines('\nAlready found {} in answer'.format(string))
        #         copy_of_all_string_perms.pop(copy_of_all_string_perms.index(string))
        # if len(copy_of_all_string_perms) != 0:
        #     for leftover_perms in copy_of_all_string_perms:
        #         if leftover_perms not in answer:
        #             answer = answer + leftover_perms
        #             # LOG.writelines('\nAdding lefover: {}'.format(leftover_perms))
        # # LOG.writelines('\nFinal answer: {}'.format(answer))
        return shortest_solution.string


def shove_in(current_string, overlapped_stuff, index_at_current_string, next_string):
    # LOG.writelines('\nFrom current string: "{}", trying to shove in next string: "{}" with this overlap: "{}"'.format(
    #     current_string, next_string, overlapped_stuff))

    if index_at_current_string != 0:
        # LOG.writelines('\nOverlap between base string: "{}" and next string: "{}" occurs at START of next string'.format(
        #     current_string, next_string))
        unique_next_string = next_string[len(overlapped_stuff):]
        # LOG.writelines('\nUnique string: {}'.format(unique_next_string))
        response = current_string + unique_next_string
    else:
        # LOG.writelines('\nOverlap between base string: "{}" and next string: "{}" occurs at END of next string'.format(
        #     current_string, next_string))
        unique_next_string = next_string[:-len(overlapped_stuff)]
        # LOG.writelines('\nUnique string: {}'.format(unique_next_string))
        response = unique_next_string + current_string

    # LOG.writelines('\nReturning: "{}"'.format(response))
    return response


def get_overlap(base_string, extra_string):
    total_overlapped = ''
    overlap = None
    index = None
    # LOG.writelines('\nLooking for overlap between {} and {}'.format(base_string, extra_string))
    for length in range(0, len(extra_string)):
        possible_overlap_1 = extra_string[length:]  # end of extra string, beginning of base string
        possible_overlap_2 = extra_string[:-length - 1]  # start of extra string, end of base string

        # LOG.writelines('\n1. Checking {} in substring of base string: {}'.format(
        #     possible_overlap_1, base_string[:len(possible_overlap_1)]))
        # base_string[:len(possible_overlap_2)])
        if possible_overlap_1 in base_string[:len(possible_overlap_1)] and possible_overlap_1 != "":
            index_1 = base_string.index(possible_overlap_1)
            # LOG.writelines('\n1. Only valid index: 0, got index: {}'.format(index_1))
            # if index_1 == 0:
            # LOG.writelines('\n1. Identified overlap "{}" at index {} in {}'.format(
            #     possible_overlap_1, index_1, base_string))
            if len(str(possible_overlap_1)) >= len(total_overlapped):
                total_overlapped = possible_overlap_1
                overlap = possible_overlap_1
                index = index_1

        # base_string[len(base_string) - len(possible_overlap_1):]
        # LOG.writelines('\n2. Checking {} in substring of base string: {}'.format(
        #     possible_overlap_2, base_string[len(base_string) - len(possible_overlap_2):]))
        if possible_overlap_2 in base_string[len(base_string) - len(possible_overlap_2):] and possible_overlap_2 != "":
            index_2 = base_string.rindex(possible_overlap_2)
            # LOG.writelines('\n2. Last valid index: {}, got index: {}'.format(len(base_string) - length, index_2))
            # if index_2 == len(base_string) - length:
            # LOG.writelines('\n2. Identified overlap "{}" at index {} in {}'.format(
            #     possible_overlap_2, index_2, base_string))
            if len(str(possible_overlap_1)) >= len(total_overlapped):
                total_overlapped = possible_overlap_2
                overlap = possible_overlap_2
                index = index_2

        # Exit out early if overlap is found
        if overlap is not None:
            return overlap, index

    # LOG.writelines('\nFailed to find overlap between base string: {} and next string: {}'.format(base_string, extra_string))
    return None, None
