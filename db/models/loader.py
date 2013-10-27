class Loader(object):

    def __init__(self):
        self.n_samples = None

    def check_n_samples(self, n):
        if self.n_samples is None:
            self.n_samples = n
        else:
            assert(self.n_samples == n)
