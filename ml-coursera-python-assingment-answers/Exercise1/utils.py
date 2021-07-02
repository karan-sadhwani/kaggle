import numpy as np
import sys
from urllib.parse import urlencode
from urllib.request import urlopen
import pickle
import json
from collections import OrderedDict
import numpy as np
import os
sys.path.append('..')
import requests

#mport submission
#import submission
requests.get('https://www-origin.coursera.org/api/', verify=False)

class SubmissionBase:

    submit_url = 'https://www-origin.coursera.org/api/' \
                 'onDemandProgrammingImmediateFormSubmissions.v1'
    save_file = 'token.pkl'

    def __init__(self, assignment_slug, part_names):
        self.assignment_slug = assignment_slug
        self.part_names = part_names
        self.login = None
        self.token = None
        self.functions = OrderedDict()
        self.args = dict()

    def grade(self):
        print('\nSubmitting Solutions | Programming Exercise %s\n' % self.assignment_slug)
        self.login_prompt()

        # Evaluate the different parts of exercise
        parts = OrderedDict()
        for part_id, result in self:
            parts[str(part_id)] = {'output': sprintf('%0.5f ', result)}
        result, response = self.request(parts)
        response = json.loads(response.decode("utf-8"))

        # if an error was returned, print it and stop
        if 'errorMessage' in response:
            print(response['errorMessage'])
            return

        # Print the grading table
        print('%43s | %9s | %-s' % ('Part Name', 'Score', 'Feedback'))
        print('%43s | %9s | %-s' % ('---------', '-----', '--------'))
        for part in parts:
            part_feedback = response['partFeedbacks'][part]
            part_evaluation = response['partEvaluations'][part]
            score = '%d / %3d' % (part_evaluation['score'], part_evaluation['maxScore'])
            print('%43s | %9s | %-s' % (self.part_names[int(part) - 1], score, part_feedback))
        evaluation = response['evaluation']
        total_score = '%d / %d' % (evaluation['score'], evaluation['maxScore'])
        print('                                  --------------------------------')
        print('%43s | %9s | %-s\n' % (' ', total_score, ' '))

    def login_prompt(self):
        if os.path.isfile(self.save_file):
            with open(self.save_file, 'rb') as f:
                login, token = pickle.load(f)
            reenter = input('Use token from last successful submission (%s)? (Y/n): ' % login)

            if reenter == '' or reenter[0] == 'Y' or reenter[0] == 'y':
                self.login, self.token = login, token
                return
            else:
                os.remove(self.save_file)

        self.login = input('Login (email address): ')
        self.token = input('Token: ')

        # Save the entered credentials
        if not os.path.isfile(self.save_file):
            with open(self.save_file, 'wb') as f:
                pickle.dump((self.login, self.token), f)

    def request(self, parts):
        params = {
            'assignmentSlug': self.assignment_slug,
            'secret': self.token,
            'parts': parts,
            'submitterEmail': self.login}

        params = urlencode({'jsonBody': json.dumps(params)}).encode("utf-8")
        f = urlopen(self.submit_url, params)
        try:
            return 0, f.read()
        finally:
            f.close()

    def __iter__(self):
        for part_id in self.functions:
            yield part_id

    def __setitem__(self, key, value):
        self.functions[key] = value


def sprintf(fmt, arg):
    """ Emulates (part of) Octave sprintf function. """
    if isinstance(arg, tuple):
        # for multiple return values, only use the first one
        arg = arg[0]

    if isinstance(arg, (np.ndarray, list)):
        # concatenates all elements, column by column
        return ' '.join(fmt % e for e in np.asarray(arg).ravel('F'))
    else:
        return fmt % arg


class Grader(SubmissionBase):
    X1 = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
    Y1 = X1[:, 1] + np.sin(X1[:, 0]) + np.cos(X1[:, 1])
    X2 = np.column_stack((X1, X1[:, 1]**0.5, X1[:, 1]**0.25))
    Y2 = np.power(Y1, 0.5) + Y1

    def __init__(self):
        part_names = ['Warm up exercise',
                      'Computing Cost (for one variable)',
                      'Gradient Descent (for one variable)',
                      'Feature Normalization',
                      'Computing Cost (for multiple variables)',
                      'Gradient Descent (for multiple variables)',
                      'Normal Equations']
        super().__init__('linear-regression', part_names)

    def __iter__(self):
        for part_id in range(1, 8):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func()
                elif part_id == 2:
                    res = func(self.X1, self.Y1, np.array([0.5, -0.5]))
                elif part_id == 3:
                    res = func(self.X1, self.Y1, np.array([0.5, -0.5]), 0.01, 10)
                elif part_id == 4:
                    res = func(self.X2[:, 1:4])
                elif part_id == 5:
                    res = func(self.X2, self.Y2, np.array([0.1, 0.2, 0.3, 0.4]))
                elif part_id == 6:
                    res = func(self.X2, self.Y2, np.array([-0.1, -0.2, -0.3, -0.4]), 0.01, 10)
                elif part_id == 7:
                    res = func(self.X2, self.Y2)
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0
