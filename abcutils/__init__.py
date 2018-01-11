"""
Initialize the abcutils.* namespace
"""
import os
import json

# Load system-wide constants
ABC_CONFIG = os.environ.get('ABC_CONFIG',
                            os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'abcconfig.json'))

CONFIG = json.load(open(ABC_CONFIG, 'r'))

from .core import *
