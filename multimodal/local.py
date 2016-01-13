"""File containg local configuration (path, etc)."""


from os.path import expanduser, join


from .lib.config import Config


CONFIG = Config()  # Initialize the configuration. This is required.

# Set useful variables
CONFIG['data-dir'] = expanduser('~/work/data/')
# Root for data sets
CONFIG['db-dir'] = join(CONFIG['data-dir'], 'db')
# To store pre-computed features
CONFIG['feat-dir'] = join(CONFIG['data-dir'], 'features')
