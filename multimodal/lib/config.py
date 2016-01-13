class NoConfigValueError(LookupError):

    def __init__(self, *args):
        super(NoConfigValueError, self).__init__(*args)
        self.message = ("No local configuration value for: '%s'." % args[0])


class Config(dict):

    def __getitem__(self, x):
        # Re-write for better error message
        try:
            return super(Config, self).__getitem__(x)
        except KeyError as e:
            raise NoConfigValueError(e.args[0])
