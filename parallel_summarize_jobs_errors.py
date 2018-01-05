#!/usr/bin/env python
#
#  The output of parallel_summarize_job.sh can be very verbose and difficult to
#  sift through to find jobs that were not correctly processed.  This script
#  scrapes the combined stderr+stdout stream of parallel_summarize_job.sh and
#  highlights the darshan log file and error generated for jobs that failed to
#  produce a summary json file.
#

import os
import sys
import re

REX_START = re.compile('UserWarning: Unhandled exception while processing (.*)$')
REX_END = re.compile('^([a-zA-z]+:.*|AssertionError)$')

filename = None
for line in open(sys.argv[1], 'r'):
    found = REX_START.search(line)
    if filename is None and found:
        filename = found.group(1)
    elif filename is not None:
        if REX_END.search(line):
            print line.strip(), filename
            filename = None
