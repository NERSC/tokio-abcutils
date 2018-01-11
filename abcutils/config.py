"""
Load system-wide constants
"""
import os
import json
ABC_CONFIG = os.environ.get('ABC_CONFIG',
                            os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'abcconfig.json'))

CONFIG = json.load(open(ABC_CONFIG, 'r'))
