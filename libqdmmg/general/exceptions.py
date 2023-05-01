'''

author: Linus Bjarne Dittmer

'''


class InvalidIntegralRequestStringException(Exception):

    def __init__(self, rq, int_class, *args):
        super().__init__(args)
        self.rq = rq
        self.int_class = int_class.lower().strip()

    def __str__(self):
        a = ""
        if self.int_class == 'elem':
            a = "elementary "
        elif self.int_class == 'comp':
            a = "composite "
        return f"The received integral request string is not a valid {a}integral descriptor." 

