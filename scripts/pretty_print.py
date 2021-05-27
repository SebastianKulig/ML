class Style:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    def print_black(self, text):
        print('{}{}{}'.format(self.BLACK, text, self.RESET))

    def print_red(self, text):
        print('{}{}{}'.format(self.RED, text, self.RESET))

    def print_green(self, text):
        print('{}{}{}'.format(self.GREEN, text, self.RESET))

    def print_yellow(self, text):
        print('{}{}{}'.format(self.YELLOW, text, self.RESET))

    def print_blue(self, text):
        print('{}{}{}'.format(self.BLUE, text, self.RESET))

    def print_magenta(self, text):
        print('{}{}{}'.format(self.MAGENTA, text, self.RESET))

    def print_cyan(self, text):
        print('{}{}{}'.format(self.CYAN, text, self.RESET))

    def print_white(self, text):
        print('{}{}{}'.format(self.WHITE, text, self.RESET))

    def print_underline(self, text):
        print('{}{}{}'.format(self.UNDERLINE, text, self.RESET))

