"""
Created December 30, 2019
Updated Jan 29, 2020

Author: Sulles

=== OVERVIEW 2.0 ===
+ SQLDataBase object is used to interact with the SQL database. It is responsible for:
    - Execute, fetch, commit, insert, and delete SQL commands for Permutation objects
+ SolverManager will organize the multi-processing of the solution. It is responsible for:
    - Queueing solution stepping for multi-processing
    - Queueing and handling all responses for the multi-processed solution step
    - Keeping track of the optimal solution path and removing all sub-optimal Permutation objects
+ Permutation object will hold all data necessary for a solution process to occur. It is responsible for:
    - Maintaining the "root_perm" string which is the work-in-progress solution
    - Maintaining a list of all other permutations that need to be added to "root_perm"
    - Housing the "overlap" and "shove_in" methods for identifying maximum overlap and updating the "root_perm"
    - Returning a list of potential equivalent solutions to SolverManager
"""

import multiprocessing
import sqlite3
from copy import copy
from datetime import datetime
from itertools import permutations
from os import unlink
from time import time


# Starting fermentation...
def step_perm(perm):
    """ This method exists to work around Pickling and Asynchronous issues """
    # print('Stepping Permutation object:\n{}'.format(str(perm)))
    response = perm.step()
    # print('Got response: {}'.format(response))
    return response


class CustomError(Exception):
    """ This is the base CustomError class object for all custom errors raised within this program """
    pass


class CustomSQLExecuteError(CustomError):
    """ This is the SQL execution-specific custom error """
    pass


class CustomSQLFetchError(CustomError):
    """ This is the SQL fetch-specific custom error """
    pass


class CustomSQLCommitError(CustomError):
    """ This is the SQL commit-specific custom error """
    pass


class CustomPermCreateError(CustomError):
    """ This is the error for creating a Permutation object from saved SQL data """
    pass


class CustomHelperFunctionError(CustomError):
    """ This Error is specific to helper functions """
    pass


def permutations_from_string_to_list(perm_string, num_of_perms):
    """ This function decodes permutations strings into a list of usable permutations for the Perm object """
    try:
        perm_list = list()
        len_of_each_perm = int(len(perm_string) / num_of_perms)
        # print('Decoding perm string: "{}", expecting {} perms'.format(perm_string, num_of_perms))
        # print('Length of each perm: {}'.format(len_of_each_perm))
        for _ in range(0, len(perm_string), len_of_each_perm):
            # print('Getting chars {} - {} in {}'.format(_, _ + len_of_each_perm, perm_string))
            perm_list.append(str(perm_string[_: _ + len_of_each_perm]))
        # print('Got perm list: {}'.format(perm_list))
        return perm_list
    except Exception as e:
        print('permutations_from_string_to_list() Failed with perm_string: {} and num_of_perms: {}\n{}'.format(
            perm_string, num_of_perms, e))
        raise CustomHelperFunctionError


def standalone_shove_in(log_ref, root_perm, overlapped_stuff, index_at_self, next_string):
    try:
        # log_ref('-- SHOVE IN --')
        # log_ref('From standalone string "{}", trying to shove in "{}", which has overlap: "{}" and index of self:{}'.format(
        #     root_perm, next_string, overlapped_stuff, index_at_self))

        if index_at_self != 0:
            # log_ref('Overlap between standalone string "{}" and other string "{}" is at START of other string'.format(
            #     root_perm, next_string))
            unique_next_string = next_string[len(overlapped_stuff):]
            # log_ref('Unique string: "{}"'.format(unique_next_string))

            root_perm = root_perm + unique_next_string
            # log_ref('Standalone string is now: "{}"'.format(root_perm))

        else:
            # log_ref('Overlap between standalone string "{}" and other string "{}" is at END of other string'.format(
            #     root_perm, next_string))
            unique_next_string = next_string[:-len(overlapped_stuff)]
            # log_ref('Unique string: "{}"'.format(unique_next_string))

            root_perm = unique_next_string + root_perm
            # log_ref('Standalone string is now: "{}"'.format(root_perm))

        return root_perm

    except Exception as e:
        print('standalone_shove_in() Failed with root_perm: {}, overlapped_stuff: {}, index_at_self: {}, '
              'next_string: {}\n{}'.format(root_perm, overlapped_stuff, index_at_self, next_string, e))
        raise CustomHelperFunctionError


def stringify(list_data):
    try:
        string = ''
        for _ in list_data:
            string += _
        return string
    except Exception as e:
        print('stringify() failed with list_data: {}\n{}'.format(list_data, e))
        raise CustomHelperFunctionError


class LogWorthy:
    def __init__(self, file_name):
        if file_name is not None:
            self.file_name = file_name
        else:
            self.file_name = None

    def _log(self, log):
        pass
        # log = '[{}] {}'.format(datetime.now(), log)
        # print(log)
        # if self.file_name is not None:
        #     with open(self.file_name, 'a') as file:
        #         file.writelines('\n' + log)


class Perm(LogWorthy):
    """ Permutation object which stores all information necessary for the Solver """

    def __init__(self, eyed, root_perm, all_other_perms, log_file_name=None):
        """
        Initializer for the sql permutation object
        :param eyed: Int identifier
        :param root_perm: Root permutation string
        :param all_other_perms: List of strings of all other permutations
        """
        assert_type(eyed, int, 'id')
        assert_type(root_perm, str, 'root_perm')
        assert_type(all_other_perms, list, 'all_other_perms')
        [assert_type(_, str, 'all_other_perms element') for _ in all_other_perms]
        self.eyed = eyed
        self.root_perm = root_perm
        self.all_other_perms = all_other_perms
        self.step_end_data = None
        LogWorthy.__init__(self, log_file_name)
        self.is_marked_for_deletion = False

    def __str__(self):
        """ This method just converts all of the information of a permutation object into a human-readable format """
        return '{header}\nID: {id} \nRoot Permutation: {root_perm} \nAll Other Permutations: {all_other_perms}\n'.format(
            header='\n=== PERMUTATION OBJECT ====', id=self.eyed, root_perm=self.root_perm,
            all_other_perms=self.all_other_perms)

    def log(self, log):
        self._log('[Perm-{}] {}'.format(self.eyed, log))
        # pass

    def step(self):
        """ This method is what will be multi-processed """
        # self.log('Got request to step!')
        response = self.find_shortest_overlap()
        # self.log('Step end')
        # self.log('Storing: {}'.format(response))
        self.step_end_data = response
        return self

    def get_step_end_data(self):
        data = copy(self.step_end_data)
        self.step_end_data = None
        return data

    def overlap(self, other_string, best_overlap_len):
        """
        Find the overlap between one permutation object and another
        :param other_string: Other permutation string that is to be compared against
        :param best_overlap_len: current highest overlap count
        :return: Returns multiple things:
            - String: The overlap found between the two
            - Int: Index of overlap in larger string
            - String: The other_string
            - Boolean: Was found to be duplicated within self
        """
        # self.log('-- OVERLAP --')

        if other_string in self.root_perm:
            print('\nCRITICAL ERROR 6: other permutation should not exist in self!')
            # self.log('\nCRITICAL ERROR 6: other permutation should not exist in self!')
            return None, None, other_string, True

        total_overlapped = ''
        overlap_1 = None
        # overlap_2 = None
        index_1 = None
        # index_2 = None
        # self.log('Looking for overlap between {} and {}'.format(self.root_perm, other_string))
        for length in range(1, len(other_string)):
            possible_overlap_1 = other_string[length:]  # end of smaller string, beginning of base string
            # possible_overlap_2 = other_string[:len(possible_overlap_1)]  # start of smaller string, end of base string

            if best_overlap_len is not None and len(possible_overlap_1) < best_overlap_len:
                # self.log('1. Possible overlap 1 "{}" length was shorter than current best overlap length: {}'.format(
                #     possible_overlap_1, best_overlap_len))
                # if len(possible_overlap_1) != len(possible_overlap_2):
                #     print('\nCRITICAL ERROR 6: Possible overlaps should be the same length!')
                #     # self.log('\nCRITICAL ERROR 6: Possible overlaps should be the same length!')
                return None, None, None, False

            # self.log('1. Checking {} in substring of longer string: {}'.format(
            #     possible_overlap_1, self.root_perm[:len(possible_overlap_1)]))
            if possible_overlap_1 == self.root_perm[:len(possible_overlap_1)] and possible_overlap_1 != "":
                index_1 = self.root_perm.index(possible_overlap_1)
                # self.log('1. Identified overlap "{}" at index {} in longer string {}'.format(possible_overlap_1,
                #                                                                              index_1, self.root_perm))
                if len(str(possible_overlap_1)) >= len(total_overlapped):
                    total_overlapped = possible_overlap_1
                    overlap_1 = possible_overlap_1

            # # self.log('2. Checking {} in substring of longer string: {}'.format(
            # #     possible_overlap_2, self.root_perm[len(self.root_perm) - len(possible_overlap_2):]))
            # if (possible_overlap_2 == self.root_perm[len(self.root_perm) - len(possible_overlap_2):]) and \
            #         (possible_overlap_2 != ""):
            #     index_2 = self.root_perm.rindex(possible_overlap_2)
            #     # self.log('2. Identified overlap "{}" at index {} in longer string {}'.format(possible_overlap_2,
            #     #                                                                              index_2, self.root_perm))
            #     if len(str(possible_overlap_1)) >= len(total_overlapped):
            #         total_overlapped = possible_overlap_2
            #         overlap_2 = possible_overlap_2

            # Exit out early if overlap is found
            # if overlap_1 is not None and overlap_2 is not None:
            #     return [overlap_1, overlap_2], [index_1, index_2], other_string, False
            if overlap_1 is not None:
                return overlap_1, index_1, other_string, False
            # if overlap_2 is not None:
            #     return overlap_2, index_2, other_string, False

        return None, None, None, False

    def shove_in(self, overlapped_stuff, index_at_self, next_string):
        # self.log('-- SHOVE IN --')
        # self.log('From current string "{}", trying to shove in "{}", which has overlap: "{}" and index of self:{}'.
        #     format(self.root_perm, next_string, overlapped_stuff, index_at_self))

        if index_at_self != 0:
            # self.log('Overlap between self "{}" and other string "{}" is at START of other string'.format(
            #     self.root_perm, next_string))
            unique_next_string = next_string[len(overlapped_stuff):]
            # self.log('Unique string: "{}"'.format(unique_next_string))

            self.root_perm = self.root_perm + unique_next_string
            # self.log('Self has new string: "{}"'.format(self.root_perm))

        else:
            # self.log('Overlap between self "{}" and other string "{}" is at END of other string'.format(
            #     self.root_perm, next_string))
            unique_next_string = next_string[:-len(overlapped_stuff)]
            # self.log('Unique string: "{}"'.format(unique_next_string))

            self.root_perm = unique_next_string + self.root_perm
            # self.log('Self has new string: "{}"'.format(self.root_perm))

    def find_shortest_overlap(self):
        """
        This method:
            - Identifies the shortest overlap possibilities
            - Explores the next shortest solution
            - Returns all potentially equivalent solutions
            - Returns solution

        All of this is returned through dictionaries of the following structure depending on the scenario:
            - key='type', value='solution' -OR- 'equal_perm' -OR- 'step_complete'
            - key='solution', value=self.root_perm
            - key='equal_perms', value=list(dict(root_perm, all_other_perms))
        """
        # Sanitize
        for perm in self.all_other_perms:
            if perm in self.root_perm:
                # self.log('- Removed duplicate')
                self.all_other_perms.pop(self.all_other_perms.index(perm))

        # Check if any permutations are left, if not, return root as solution
        if len(self.all_other_perms) == 0:
            return dict(type='solution', solution=self.root_perm)

        # self.log('Trying to find next optimal levels')
        # self.log('Starting with: "{}"'.format(self.root_perm))
        # self.log('All other perms: {}'.format(self.all_other_perms))
        all_overlaps = list()
        all_overlap_lengths = list()
        all_overlap_indexes = list()
        all_tested_perms = list()
        current_max_len = 0
        for x in range(len(self.all_other_perms)):
            overlap, index, perm, is_duplicate = self.overlap(self.all_other_perms[x], current_max_len)
            # self.log('Got overlap: {}'.format(overlap))
            # self.log('Got index: {}'.format(index))
            # self.log('Got perm: {}'.format(perm))
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
                # self.log('Got new maximum length overlap! Is length: {}'.format(current_max_len))

        if len(all_overlaps) == 0:
            return self.handle_no_optimal_solution()

        # self.log('Got all overlaps: "{}"'.format(all_overlaps))
        number_of_maximum_overlaps = all_overlap_lengths.count(current_max_len)
        # self.log('There are {} occurrences of maximum length overlap'.format(number_of_maximum_overlaps))

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
            # self.log('Got optimal overlap: "{}", has index: {}. Base string: "{}" has overlap "{}" with "{}"'.format(
            #     optimal_overlap[-1], optimal_index[-1], self.root_perm, all_overlaps[optimal_index[-1]],
            #     optimal_combination_perm[-1]))
            # remove max_length from all overlap lengths
            all_overlap_lengths[optimal_index[-1]] = 0

        # Keep temporary list copy of root_perm
        temp_root_perm = copy(self.root_perm)
        temp_list_of_all_other_perms = copy(self.all_other_perms)

        # Update this perm with first maximum overlap possibility
        optimal_perm = optimal_combination_perm.pop(0)
        self.shove_in(optimal_overlap.pop(0), optimal_overlap_index.pop(0), optimal_perm)
        self.all_other_perms.pop(self.all_other_perms.index(optimal_perm))  # Remove from list of remaining perms
        if len(self.all_other_perms) == 0:
            return dict(type='solution', solution=self.root_perm)

        # Return all other maximum overlap options with all data necessary to create a new Perm object
        if len(optimal_overlap) == 0:
            return dict(type='step_complete')
        else:
            all_equal_root_perms = list()
            all_equal_all_other_perms = list()
            for x in range(len(optimal_overlap)):
                # Get next equal root perm using standalone shove-in calculator
                all_equal_root_perms.append(standalone_shove_in(self.log, temp_root_perm, optimal_overlap[x],
                                                                optimal_overlap_index[x], optimal_combination_perm[x]))
                # Get list of all other perms excluding the perm that was shoved in in this previous step ^
                all_equal_all_other_perms.append(
                    # [optimal_combination_perm[:x] + optimal_combination_perm[x + 1:]])
                    [_ for _ in temp_list_of_all_other_perms if _ != optimal_combination_perm[x]])
            return dict(type='equal_perms', equal_perms=[
                dict(root_perm=all_equal_root_perms[_], all_other_perms=all_equal_all_other_perms[_])
                for _ in range(len(all_equal_root_perms))])

    def handle_no_optimal_solution(self):
        temp_root_perm = copy(self.root_perm)
        all_other_perms = copy(self.all_other_perms)

        self.root_perm += self.all_other_perms.pop(0)

        all_equal_root_perms = list()
        all_equal_all_other_perms = list()
        for x in range(len(all_other_perms)):
            # Get next equal root perm using standalone shove-in calculator
            all_equal_root_perms.append(temp_root_perm + all_other_perms[x])
            # Get list of all other perms excluding the perm that was shoved in in this previous step ^
            all_equal_all_other_perms.append(
                all_other_perms[:x] + all_other_perms[x + 1:])
            # [_ for _ in all_other_perms if _ != ])
        return dict(type='equal_perms', equal_perms=[
            dict(root_perm=all_equal_root_perms[_], all_other_perms=all_equal_all_other_perms[_])
            for _ in range(len(all_equal_root_perms))])


class SQLDataBase(LogWorthy):
    def __init__(self, log_file=None, database_file='suleyman.db'):
        """
        Initializer for sql database handler
        """
        self.name = 'SQLDataBase'
        self.database_file = database_file
        LogWorthy.__init__(self, log_file)

        # Created db, open connection, and create cursor
        open(self.database_file, 'w').close()
        self.connection = sqlite3.connect(self.database_file)
        self.cursor = self.connection.cursor()
        # self.tracking = dict(store_perm=0, add_to_backlog=0, add_multiple_to_backlog=0, add_solution=0, get_perm=0,
        #                      get_all_perm_data=0, get_next_backlog_perm=0, get_all_backlog_data=0, get_all_solutions=0,
        #                      delete_perm=0, delete_backlog_perm=0, delete_multiple_backlog_perms=0,
        #                      delete_all_backlog=0, get_multiple_backlog_perms=0)

        # Create database tables
        try:
            self.execute("""
                CREATE TABLE perm_objects
                (id int, root_perm text, all_other_perms text, root_len int, num_of_remaining_perms int)
                """)
            self.execute("""
                CREATE TABLE backlog
                (id int, root_perm text, all_other_perms text, root_len int, num_of_remaining_perms int)
                """)
            self.execute(""" CREATE TABLE solutions (solution text) """)
            # self.log('Created perm_objects, backlog, and solutions tables in SQL database')
        except Exception as e:
            print('\nCRITICAL ERROR 0: Error raised during SQL Table creation: {}'.format(e))
            # self.log('\nCRITICAL ERROR 0: Error raised during SQL Table creation: {}'.format(e))
            raise

    def log(self, log):
        self._log('[{}] {}'.format(self.name, log))
        # pass

    """ === SQL INTERFACE === """

    def execute(self, string):
        try:
            self.cursor.execute(string)
            self.connection.commit()
        except sqlite3.OperationalError as e:
            print('\nCRITICAL ERROR 2: Incorrect SQL command: "{}"\n{}'.format(string, e))
            # self.log('\nCRITICAL ERROR 2: Incorrect SQL command: "{}"\n{}'.format(string, e))
            raise CustomSQLExecuteError
        except Exception as e:
            print('\nCRITICAL ERROR 2: Failed to execute SQL command: "{}"\n{}'.format(string, e))
            # self.log('\nCRITICAL ERROR 2: Failed to execute SQL command: "{}"\n{}'.format(string, e))
            raise CustomSQLExecuteError

    def fetch(self):
        try:
            return self.cursor.fetchall()
        except Exception as e:
            print('\nCRITICAL ERROR 3: Failed to fetch all!\n{}'.format(e))
            # self.log('\nCRITICAL ERROR 3: Failed to fetch all!\n{}'.format(e))
            raise CustomSQLFetchError

    def commit(self):
        try:
            self.connection.commit()
        except Exception as e:
            print('\nCRITICAL ERROR 4: Failed to commit!\n{}'.format(e))
            # self.log('\nCRITICAL ERROR 4: Failed to commit!\n{}'.format(e))
            raise CustomSQLCommitError

    """ === SETTERS === """

    def store_perm(self, perm, delete=True):
        """ This method adds a permutation object to the SQL database """
        # self.tracking['store_perm'] += 1
        # self.log('Storing permutation object ID: {}'.format(perm.eyed))
        all_other_perms_as_str = ''
        for _ in perm.all_other_perms:
            all_other_perms_as_str += _
        self.execute("""
            INSERT INTO perm_objects (id, root_perm, all_other_perms, root_len, num_of_remaining_perms)
            VALUES ('{id}', '{root_perm}', '{all_other_perms}', '{root_len}', '{num_of_remaining_perms}')
            """.format(id=perm.eyed, root_perm=perm.root_perm, all_other_perms=all_other_perms_as_str,
                       root_len=len(perm.root_perm), num_of_remaining_perms=len(perm.all_other_perms)))
        if delete:
            # self.log('Deleting permutation object with ID: {}'.format(perm.eyed))
            del perm

    def add_to_backlog(self, eyed, perm_data):
        """
        This method will add a permutation object to the backlog in the SQL database
        :param eyed: Identification number for backlog permutation
        :param perm_data: all data required to create a Permutation object in the form of a dictionary.
        """
        # self.tracking['add_to_backlog'] += 1
        # self.log('Adding permutation data to backlog: {}'.format(perm_data))
        all_other_perms_as_str = ''
        for _ in perm_data['all_other_perms']:
            all_other_perms_as_str += _
        self.execute("""
            INSERT INTO backlog (id, root_perm, all_other_perms, root_len, num_of_remaining_perms)
            VALUES ('{id}', '{root_perm}', '{all_other_perms}', '{root_len}', '{num_of_remaining_perms}')
            """.format(id=eyed, root_perm=perm_data['root_perm'],
                       all_other_perms=stringify(perm_data['all_other_perms']),
                       root_len=len(perm_data['root_perm']), num_of_remaining_perms=len(perm_data['all_other_perms'])))

    def add_multiple_to_backlog(self, id_list, perm_data_list):
        """
        This method will add a permutation object to the backlog in the SQL database
        :param id_list: List of Identification numbers for backlog permutation
        :param perm_data_list: List of all data required to create a Permutation object in the form of a dictionary.
        """
        # self.tracking['add_multiple_to_backlog'] += 1
        assert len(id_list) == len(perm_data_list), 'Must have matching lengths of ids and permutation data!'
        execution_statement = \
            """ INSERT INTO backlog (id, root_perm, all_other_perms, root_len, num_of_remaining_perms) 
                VALUES """
        # self.log('Requesting to add this to backlog:')
        # for _ in perm_data_list:
        # self.log(_)
        for x in range(len(id_list)):
            execution_statement += \
                """\t('{id}', '{root_perm}', '{all_other_perms}', '{root_len}', '{num_of_remaining_perms}'){comma} """.format(
                    id=id_list[x], root_perm=perm_data_list[x]['root_perm'],
                    all_other_perms=stringify(perm_data_list[x]['all_other_perms']),
                    root_len=len(perm_data_list[x]['root_perm']),
                    num_of_remaining_perms=len(perm_data_list[x]['all_other_perms']),
                    comma=',' if x != len(id_list) - 1 else '')
        self.execute(execution_statement)
        # self.log('Added {} elements to backlog'.format(len(id_list)))

    def add_solution(self, solution):
        """ This method simply stores a solution in the SQL database """
        # self.tracking['add_solution'] += 1
        self.execute(""" INSERT INTO solutions (solution) VALUES ('{solution}') """.format(solution=solution))

    """ === GETTERS === """

    def get_perm(self, eyed=None):
        """
        The method will find the next permutation object with shortest combination of (root perm length and
        all other perms), POP that permutation object from the sql db, and return the data as a permutation object
        """
        # self.tracking['get_perm'] += 1
        if eyed is not None:
            # Select matching permutation object
            self.execute("""
                SELECT * FROM perm_objects WHERE id='{}'
                """.format(eyed))
        else:
            # Select all permutation objects
            self.execute(""" SELECT * FROM perm_objects """)

        # Store fetch
        fetch = self.fetch()[0]
        # self.log('Got perm object: {}'.format(fetch))

        # Remove from database
        self.delete_perm(eyed)

        # Create and return permutation object
        try:
            perm = Perm(fetch[0], fetch[1], permutations_from_string_to_list(fetch[2], fetch[4]))
            return perm
        except Exception as e:
            print('\nCRITICAL ERROR 5: Failed to automatically create a Permutation object from SQL data!\n{}'.
                  format(e))
            # self.log('\nCRITICAL ERROR 5: Failed to automatically create a Permutation object from SQL data!\n{}'.
            #     format(e))
            raise CustomPermCreateError

    def get_all_perm_data(self):
        # self.tracking['get_all_perm_data'] += 1
        self.execute(""" SELECT * FROM perm_objects """)
        return self.fetch()

    def get_next_backlog_perm(self):
        """
        This method will find the next backlog permutation object and return the data as a permutation object
        # :param eyed: Identification number for backlog permutation
        :return: Permutation object that matches the ID provided
        """
        # self.tracking['get_next_backlog_perm'] += 1
        # Select matching permutation object
        self.execute(""" SELECT * FROM backlog """)

        # Store fetch
        fetch = self.fetch()[0]
        # self.log('Got perm object: {}'.format(fetch))

        # Remove from database
        self.delete_backlog_perm(fetch[0])

        # Create and return permutation object
        try:
            perm = Perm(fetch[0], fetch[1], permutations_from_string_to_list(fetch[2], fetch[4]))
            return perm
        except Exception as e:
            print('\nCRITICAL ERROR 5: Failed to automatically create a Permutation object from SQL data:\n"{}"\n{}'). \
                format(fetch, e)
            # self.log('\nCRITICAL ERROR 5: Failed to automatically create a Permutation object from SQL data:\n"{}"\n{}').\
            #     format(fetch, e)
            raise CustomPermCreateError

    def get_multiple_backlog_perms(self, num):
        """
        This method will get the next 'num' of backlog entries
        :return: List of 'num' of Permutation objects (if backlog permits, or fewer)
        """
        # self.tracking['get_multiple_backlog_perms'] += 1
        # self.tracking['get_next_backlog_perm'] += 1
        # Select matching permutation object
        self.execute(""" SELECT * FROM backlog """)

        # Get fetch_list of all permutation data
        fetch_list = self.fetch()
        # self.log('Got {} perm objects from backlog'.format(len(fetch_list)))

        # Remove all retrieved data from database
        perm_ids_to_be_removed = list()
        for fetch in fetch_list:
            perm_ids_to_be_removed.append(fetch[0])
        if len(perm_ids_to_be_removed) > 1:
            self.delete_multiple_backlog_perms(perm_ids_to_be_removed)
        elif len(perm_ids_to_be_removed) != 0:
            self.delete_backlog_perm(perm_ids_to_be_removed[0])
        else:
            return list()

        # Create and return permutation object
        try:
            perm_list = list()
            for fetch in fetch_list:
                perm_list.append(Perm(fetch[0], fetch[1], permutations_from_string_to_list(fetch[2], fetch[4])))
            return perm_list
        except Exception as e:
            print('\nCRITICAL ERROR 5: Failed to automatically create a Permutation object from SQL data:\n"{}"\n{}'). \
                format(fetch_list, e)
            # self.log(
            #     '\nCRITICAL ERROR 5: Failed to automatically create a Permutation object from SQL data:\n"{}"\n{}'). \
            #     format(fetch_list, e)
            raise CustomPermCreateError

    def get_all_backlog_data(self):
        # self.tracking['get_all_backlog_data'] += 1
        self.execute(""" SELECT * FROM backlog """)
        return self.fetch()

    def get_all_solutions(self):
        # self.tracking['get_all_solutions'] += 1
        self.execute(""" SELECT * FROM solutions """)
        return self.fetch()

    def get_backlog_length(self):
        self.execute(""" SELECT COUNT(*) FROM backlog """)
        return self.fetch()

    """ === REMOVERS === """

    def delete_perm(self, eyed):
        # self.tracking['delete_perm'] += 1
        self.execute("""
            DELETE FROM perm_objects WHERE id='{}'
            """.format(eyed))
        # self.log('Deleted perm object with id: {}'.format(eyed))

    def delete_backlog_perm(self, eyed):
        # self.tracking['delete_backlog_perm'] += 1
        self.execute("""
            DELETE FROM backlog WHERE id='{}'
            """.format(eyed))
        # self.log('Deleted backlog perm object with id: {}'.format(eyed))

    def delete_multiple_backlog_perms(self, id_list):
        # self.tracking['delete_multiple_backlog_perms'] += 1
        self.execute("""
            DELETE FROM backlog WHERE id IN {}""".format(tuple(id_list)))
        # self.log('Deleted backlog perm objects with ids: {}'.format(id_list))

    def delete_all_backlog(self):
        # self.tracking['delete_all_backlog'] += 1
        self.execute(""" DELETE FROM backlog """)
        # self.log('Deleted entire backlog!')

    def enforce_backlog_fitness(self, num_of_remaining_perms, root_len):
        # self.execute("""
        #     CREATE TABLE backlog
        #     (id int, root_perm text, all_other_perms text, root_len int, num_of_remaining_perms int)
        #     """)
        self.execute("""
            DELETE FROM backlog WHERE num_of_remaining_perms = '{}' AND root_len > '{}'""".format(
            num_of_remaining_perms, root_len))

    """ === CLEAN-UP === """

    def clean_up(self):
        """ Clean-up method """
        graceful = True
        leftovers = self.get_all_perm_data()
        if len(leftovers) != 0:
            # self.log('Clean-up found {} permutation objects that were not deleted!'.format(len(leftovers)))
            graceful = False

        leftovers = self.get_all_backlog_data()
        if len(leftovers) != 0:
            # self.log('Clean-up found {} backlog objects that were not deleted!'.format(len(leftovers)))
            graceful = False

        # if graceful:
        # self.log('Gracefully closed SQL database!')
        # else:
        # self.log('WARNING: Ungracefully closed SQL database!')

        self.connection.close()
        unlink(self.database_file)

        # print(self.tracking)


class SolverManager(LogWorthy):
    """
    The SolverManager object is responsible for:
        - Keeping track of the optimal solution path
        - Maintaining the list of active Permutation objects that may reach the solution
        - Maintaining the backlog and purging sub-optimal potential solutions
    """

    def __init__(self, initial_unique_perm_set, log_file=None):
        self.name = 'SolverManager'
        self.log_file = log_file
        LogWorthy.__init__(self, log_file)
        self.stop = False
        self.tracker = dict(delete_unfit_active_perms=0, delete_unfit_backlog_perms=0,
                            delete_duplicates_active=0, delete_duplicates_backlog=0, process_to_be_added_to_backlog=0)
        self.time_tracker = dict(add_perm_to_pool=0, wait_for_perm_processing=0, response_handling=0,
                                 delete_perms_marked_for_deletion=0, process_to_be_added_to_backlog=0,
                                 do_multiprocess_step=0, update_best_solution_path=0, delete_unfit_perms=0,
                                 fill_active_perms=0)

        # SQL and tracking
        self.database = SQLDataBase()
        # Solution path works as follows:
        #   Key: Len of all_other_perms
        #   Value: Len of root_perm
        # If any permutation has a len of remaining perms, it is expected to have the same or lower root_perm len
        self.solution_path = dict()
        self.perm_id_counter = 0
        self.solutions = list()
        self.to_be_added_to_backlog = list()

        # Queueing and multiprocessing
        self.active_perms = list()
        self.responses = list()
        self.add_to_backlog_process = None

        # Init the first permutation sets
        self.create_initial_perm_set(initial_unique_perm_set)
        self.update_best_solution_path()

        # # self.log('Number of CPUs detected: {}'.format(multiprocessing.cpu_count()))
        self.num_of_solvers = 2 * multiprocessing.cpu_count()
        self.solver_list = list()
        self.solver_pool = multiprocessing.Pool(self.num_of_solvers)
        # self.log('Created solver pool of size: {}'.format(self.num_of_solvers))

        # self.log('Initialization complete')

    def solve(self):
        """ This method handles the high-level algorithm solution """
        while len(self.active_perms) > 0 or len(self.database.get_all_backlog_data()) > 0:
            start_time = time()
            # self.log('Doing do_multiprocess_step')
            self.do_multiprocess_step()
            self.time_tracker['do_multiprocess_step'] += time() - start_time

            start_time = time()
            # self.log('Doing update_best_solution_path')
            self.update_best_solution_path()
            self.time_tracker['update_best_solution_path'] += time() - start_time

            start_time = time()
            # self.log('Doing delete_unfit_perms')
            self.delete_unfit_perms()
            self.time_tracker['delete_unfit_perms'] += time() - start_time

            # self.delete_duplicates()

            start_time = time()
            # self.log('Doing fill_active_perms')
            self.fill_active_perms()
            self.time_tracker['fill_active_perms'] += time() - start_time

            # for key in self.time_tracker.keys():
            #     self.time_tracker[key] = round(self.time_tracker[key], 2)
            # self.log('Time tracker results: {}'.format(self.time_tracker))
            # for key in self.time_tracker.keys():
            #     self.time_tracker[key] = 0

        # TODO: Return solution here!
        return self.get_shortest_solution()

    def get_shortest_solution(self):
        # self.log('All solutions:')
        # for _ in self.solutions:
        # self.log(_)
        lengths = [len(_) for _ in self.solutions]
        return self.solutions[lengths.index(min(lengths))]

    def update_best_solution_path(self):
        """ This method updates a dictionary that tracks the best known solution """
        for perm in self.active_perms:
            root_perm_len = len(perm.root_perm)
            len_of_all_other_perms = len(perm.all_other_perms)
            # If this is the first permutation with this specific number of remaining permutations...
            if len_of_all_other_perms not in self.solution_path.keys():
                # Add it to solution path
                self.solution_path[len_of_all_other_perms] = root_perm_len
            # Else if the existing solution path with this number of remaining permutations has a shorter root_perm len,
            elif self.solution_path[len_of_all_other_perms] > root_perm_len:
                # Update solution path with shorter root_perm len
                self.solution_path[len_of_all_other_perms] = root_perm_len
        # self.log('Updated solution path: {}'.format(self.solution_path))

    def delete_unfit_perms(self):
        """ This method will call helper functions to delete active and backlog perms that are not optimal solution """
        self.delete_unfit_active_perms()
        self.delete_unfit_backlog_perms()

    def delete_unfit_active_perms(self):
        """ This method deletes any active Permutation objects that are not the fittest """
        for perm in self.active_perms:
            if len(perm.root_perm) > self.solution_path[len(perm.all_other_perms)]:
                # self.log('Deleting Perm with root length of {} is longer than solution path: {}'.format(
                #     len(perm.root_perm), self.solution_path[len(perm.all_other_perms)]))
                self.active_perms.remove(perm)
                self.tracker['delete_unfit_active_perms'] += 1

    def delete_unfit_backlog_perms(self):
        """ This method deletes any backlog Permutation objects that are not the fittest """
        if self.database.get_backlog_length() != 0:
            for num_of_remaining_perms, root_len in self.solution_path.items():
                self.database.enforce_backlog_fitness(num_of_remaining_perms, root_len)

        # backlog_data = self.database.get_all_backlog_data()
        # # self.log('All backlog data:\n{}'.format(backlog_data))
        # marked_for_deletion = list()
        # for perm_data in backlog_data:
        #     num_of_all_other_perms = perm_data[-1]
        #     len_of_root_perm = perm_data[-2]
        #     try:
        #         if self.solution_path[num_of_all_other_perms] < len_of_root_perm:
        #             # self.log('Identified unfit perm! ID: {}'.format(perm_data[0]))
        #             marked_for_deletion.append(perm_data[0])
        #     except KeyError:
        #         # If solution path already has solution with fewer remaining keys, remove
        #         if min(self.solution_path.keys()) < num_of_all_other_perms:
        #             # self.log('Identified unfit perm! ID: {}'.format(perm_data[0]))
        #             marked_for_deletion.append(perm_data[0])
        #         else:
        #             # self.log('(*) Adding new solution path from backlog?!')
        #             self.solution_path[num_of_all_other_perms] = len_of_root_perm
        #             # self.log('New solution path: {}'.format(self.solution_path))
        #
        # self.tracker['delete_unfit_backlog_perms'] += len(marked_for_deletion)
        # if len(marked_for_deletion) > 1:
        #     self.database.delete_multiple_backlog_perms(marked_for_deletion)
        #     # self.log('Found and deleted {} unfit backlog perms'.format(len(marked_for_deletion)))
        # elif len(marked_for_deletion) == 1:
        #     self.database.delete_backlog_perm(marked_for_deletion[0])
        #     # self.log('Deleted one unfit backlog perm')

    def delete_duplicates(self):
        unique_roots = list()
        # Ensuring uniqueness of active perms
        for perm in self.active_perms:
            if perm.root_perm not in unique_roots:
                unique_roots.append(perm.root_perm)
            else:
                # self.log('Deleting duplicate perm')
                self.active_perms.remove(perm)
                self.tracker['delete_duplicates_active'] += 1
        # Ensuring uniqueness of backlog perms
        duplicated_backlog_ids = list()
        all_backlog_data = self.database.get_all_backlog_data()
        if len(all_backlog_data) == 0:
            return
        for data in all_backlog_data:
            if data[1] not in unique_roots:
                unique_roots.append(data[1])
            else:
                duplicated_backlog_ids.append(data[0])
                self.tracker['delete_duplicates_backlog'] += 1
        if len(duplicated_backlog_ids) > 1:
            # self.log('Deleting duplicated backlog ids: {}'.format(duplicated_backlog_ids))
            self.database.delete_multiple_backlog_perms(duplicated_backlog_ids)
        elif len(duplicated_backlog_ids) != 0:
            # self.log('Deleting duplicated backlog ids: {}'.format(duplicated_backlog_ids))
            self.database.delete_backlog_perm(duplicated_backlog_ids[0])
        else:
            # self.log('No duplicated backlog ids found')
            pass

    def fill_active_perms(self):
        """ This method will add as many backlog Permutations objects to the active perms list as possible """
        num_of_available_slots = self.num_of_solvers - len(self.active_perms)
        for perm in self.database.get_multiple_backlog_perms(num_of_available_slots):
            self.active_perms.append(perm)

    def do_multiprocess_step(self):
        """ This method performs the multi-processing step """
        try:
            # self.log('Starting {} async Solvers'.format(self.num_of_solvers))
            # running_unique_roots = list()
            start_time = time()
            # self.log('Doing add_perm_to_pool')
            results = list()
            for perm in self.active_perms:
                results.append(self.solver_pool.apply_async(step_perm, args=(perm,)))
            self.time_tracker['add_perm_to_pool'] += time() - start_time

            # self.log('Waiting for all multi-processed results')
            indx = 0
            for r in results:
                start_time = time()
                # self.log('Doing wait_for_perm_processing')
                r.wait()
                self.time_tracker['wait_for_perm_processing'] += time() - start_time

                start_time = time()
                # self.log('Doing response_handling')
                self.active_perms[indx] = r.get()

                # # Do running uniqueness check
                # if self.active_perms[indx].root_perm in running_unique_roots:
                #     # Mark for deletion and break out of for loop if found to be duplicate
                #     self.active_perms[indx].is_marked_for_deletion = True
                #     indx += 1
                #     break
                # # Append root to unique list if unique
                # running_unique_roots.append(self.active_perms[indx].root_perm)

                # self.log('Updated Perm: {}'.format(str(self.active_perms[indx])))
                response = self.active_perms[indx].get_step_end_data()
                if type(response) is not dict:
                    print('(*) Ignoring non-dict response: {}, is type: {}'.format(response, type(response)))
                elif response['type'] == 'solution':
                    # self.log('Got solution of length: {}'.format(len(response['solution'])))
                    self.solutions.append(response['solution'])
                    self.active_perms[indx].is_marked_for_deletion = True
                elif response['type'] == 'equal_perms':
                    # self.log('Got {} more potential equal solutions'.format(len(response['equal_perms'])))
                    for data in response['equal_perms']:
                        # # Only add backlog root_perm if not in running unique root_perm list
                        # if data['root_perm'] not in running_unique_roots:
                        self.to_be_added_to_backlog.append(data)
                elif response['type'] != 'step_complete':
                    # self.log('Unexpected response received! Got: {}'.format(response))
                    pass
                indx += 1
                self.time_tracker['response_handling'] += time() - start_time

            start_time = time()
            # self.log('Doing delete_perms_marked_for_deletion')
            for perm in self.active_perms:
                if perm.is_marked_for_deletion:
                    # self.log('Trying to delete perm id: {}'.format(perm.eyed))
                    self.active_perms.remove(perm)
            self.time_tracker['delete_perms_marked_for_deletion'] += time() - start_time

            start_time = time()
            # self.log('Doing process_to_be_added_to_backlog')
            if self.add_to_backlog_process is not None:
                self.add_to_backlog_process.wait()
            self.add_to_backlog_process = self.solver_pool.apply_async(self.process_to_be_added_to_backlog)
            # self.process_to_be_added_to_backlog()
            self.time_tracker['process_to_be_added_to_backlog'] += time() - start_time

        except Exception as e:
            print('\nCRITICAL ERROR 7: Multiprocessing solution failed!')
            # self.log('\nCRITICAL ERROR 7: Multiprocessing solution failed!')
            self.solver_pool.terminate()
            self.clean_up()
            raise e

        # self.log('End of multiprocess step')

    def log(self, log):
        self._log('[{}] {}'.format(self.name, log))
        # pass

    def create_initial_perm_set(self, perm_set):
        for x in range(0, len(perm_set)):
            self.perm_id_counter += 1
            self.active_perms.append(
                Perm(self.perm_id_counter, perm_set[x], perm_set[:x] + perm_set[x + 1:], self.log_file))

    def process_to_be_added_to_backlog(self):
        if len(self.to_be_added_to_backlog) == 0:
            return
        new_ids = list(range(len(self.to_be_added_to_backlog)))
        self.perm_id_counter += len(new_ids)
        # # unique_additions = list()
        # for _ in range(len(self.to_be_added_to_backlog)):
        #     # if self.to_be_added_to_backlog[_] not in unique_additions:
        #     self.perm_id_counter += 1
        #     new_ids.append(self.perm_id_counter)
        #     # unique_additions.append(self.to_be_added_to_backlog[_])
        #     # else:
        #     #     self.tracker['process_to_be_added_to_backlog'] += 1
        num_of_entries_to_be_added_to_backlog = len(self.to_be_added_to_backlog)
        try:
            # self.log('Adding {} entries to backlog'.format(len(self.to_be_added_to_backlog)))
            self.database.add_multiple_to_backlog(new_ids, self.to_be_added_to_backlog)
        except MemoryError as e:
            print('Failed to add {} entries to the backlog'.format(num_of_entries_to_be_added_to_backlog))
            raise e
        self.to_be_added_to_backlog = list()

    def clean_up(self):
        self.database.delete_all_backlog()
        # self.log('Deletion tracking results from uniqueness checks:\n{}'.format(self.tracker))
        self.database.clean_up()
        # self.log('Time tracker results: {}'.format(self.time_tracker))


def assert_type(obj, obj_type, obj_name):
    """ Helper function for assertions """
    assert type(obj) is obj_type, '"{}" is not type {}! Got: {}'.format(obj_name, obj_type, type(obj))


def main(input_list):
    answer = ''

    perm_list = list()
    all_permutations = list(permutations(input_list))
    unique_permutations = list()
    # Get unique permutations
    for perm in all_permutations:
        if perm not in unique_permutations:
            unique_permutations.append(perm)
    # Generate all permutation objects
    for perm in unique_permutations:
        string = ''.join([str(_) for _ in perm])
        perm_list.append(string)

    # print('Got all unique permutations: {}'.format(perm_list))

    open('log_output.txt', 'w').close()
    solver_manager = SolverManager(perm_list, log_file='log_output.txt')

    try:
        answer = solver_manager.solve()
    except CustomError:
        pass
    except Exception as e:
        print('\nCRITICAL ERROR 1: Error raised during normal algorithm execution')
        raise e
    finally:
        solver_manager.clean_up()

    # # UNIT TEST 1
    # # SQLDatabase successfully add and delete a Perm object
    # perm_list = list()
    # all_permutations = list(permutations(input_list))
    # unique_permutations = list()
    # # Get unique permutations
    # for perm in all_permutations:
    #     if perm not in unique_permutations:
    #         unique_permutations.append(perm)
    # # Generate all permutation objects
    # for perm in unique_permutations:
    #     string = ''.join([str(_) for _ in perm])
    #     perm_list.append(string)
    # solver_manager = SolverManager(perm_list, log_file=None)
    # perm_data = dict(root_perm=solver_manager.active_perms[0].root_perm, all_other_perms=solver_manager.active_perms[0].all_other_perms)
    # solver_manager.database.add_to_backlog(0, perm_data)
    # print(solver_manager.database.get_all_backlog_data())
    # solver_manager.database.delete_backlog_perm(0)
    # print(solver_manager.database.get_all_backlog_data())

    # Return Answer
    return answer


def suleyman(input_list):
    return main(input_list)


if __name__ == "__main__":
    print('Testing started')
    solution = main([1, 2, 3, 4, 5])
    print('Got solution of len: {}\n{}'.format(len(solution), solution))
