import builtins
import datetime

from library.code_patterns import Singleton


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NORMAL = ''


def add_colour(text_to_print: [str, list, tuple], type_: str, keywords: [str, list, tuple] = None) -> str:
    """
    Add print-friendly colour code to string
    :param text_to_print: The text to print
    :param type_: One of the types defined in Bcolors class
    :param keywords: If included, add colour only to those keywords
    :return: A ready-to-print string
    """
    if isinstance(text_to_print, list):
        text_to_print = " ".join(text_to_print)
    if keywords is None:
        text_to_print = getattr(Bcolors, type_) + text_to_print + Bcolors.ENDC
    else:
        if isinstance(keywords, str):
            keywords = [keywords]
        for keyword in keywords:
            text_to_print = text_to_print.replace(keyword, getattr(Bcolors, type_) + keyword + Bcolors.ENDC)

    return text_to_print


class ConditionalPrinter(Singleton):
    def __init__(self, execution_verbosity: int = 9):
        self.execution_verbosity = execution_verbosity

    def print(self, *args, verbosity: int = None, type_: str = 'NORMAL', keywords: [str, list, tuple] = None):
        """"
        Print method designed to overcharge the builtin print.
        This method has been conceived to be invoked not directly as a class method but from the conditional_print function.

        If none optional parameters are issued, works exactly the same as builtin print.
        :param verbosity: If indicated, execute print only if verbosity >= execution_verbosity.
        :param type_: If indicated, print text in the indicated color. Available color codes are included in Bcolors class.
        :param keywords: If indicated, add colour only to specified keywords.
        If
        """
        if verbosity is not None and self.execution_verbosity < verbosity:
            pass
        else:
            args = [arg() if callable(arg) else arg for arg in args]
            time = str(datetime.datetime.now()).split('.')[0]
            verbosity = 0 if verbosity is None else verbosity
            prefix = " " * verbosity + ">  "
            text_to_print = time + ' ' + (prefix if verbosity > 0 else '') + add_colour(
                ' '.join([str(a) for a in args]), type_, keywords)
            builtins.print(text_to_print)


Printer = ConditionalPrinter()


def conditional_print(*args, verbosity=None, type_='NORMAL', keyword=None, set_execution_verbosity: int = None):
    if set_execution_verbosity is not None:
        Printer.execution_verbosity = set_execution_verbosity
        if not len(args) == 0:
            print(add_colour(
                'Warning: conditional_print function has been called with set_execution_verbosity'
                'but *args was no empty. args: {}'.format(args),
                keywords='Warning:', type_='WARNING'))
    else:
        Printer.print(*args, verbosity=verbosity, type_=type_, keywords=keyword)


class ProgressBar:
    """
    ProgressBar object to keep count of progress and optional parameters.
    Total possible progress must be specified at initialization.
    """

    def __init__(self, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 100,
                 print_each: int = 1, fill: str = '█'):
        """
        Initialize the class.
        :param total: Required - total progress required to complete the bar
        :param prefix: Prefix string
        :param suffix: Suffix string
        :param decimals: Positive number of decimals in percent complete
        :param length: Character length of bar
        :param print_each: Print progress each X iterations
        :param fill: Bar fill character
        """
        self.progress = 0
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.print_each = print_each
        self.fill = fill

    def print(self, add: int = 1):
        """
        Add 1 to bar progress and print the bar.
        :type add: increasement of the progress
        :return: None
        """
        self.progress += add
        if self.progress % self.print_each == 0:
            print_progres_bar(self.progress, self.total, prefix=self.prefix, suffix=self.suffix, decimals=self.decimals,
                              length=self.length, fill=self.fill)

    def fixed_print(self, progress: int, total: int = None):
        """
        Print progress indicating a specific progress rather than the one recorded by the class.
        Total progress expressed can also be declared splicitly.
        This print won't add one step to the class progress.

        :param progress: Specific progress to print on the bar
        :param total: Optional. Total progress to complete the bar
        :return: None
        """
        print_progres_bar(progress, self.total if total is None else total)


def print_progres_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))  # , end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()