#!/usr/bin/env python
"""
Convert a collection of individual json summary files from summarize_job.py
--json into a single, normalized CSV file.  This CSV is what is typically
loaded into subsequent analysis.
"""

import os
import sys
import json
import argparse
import datetime
import pandas

def _normalize_job_summaries():
    parser = argparse.ArgumentParser(
        description='normalize a collection of summarize_job.py output jsons')
    parser.add_argument('file', type=str, nargs='+',
                        help='summary output json file(s) to normalize')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='output file name (or "-" for stdout)')
    args = parser.parse_args()

    # Load each summary json and populate a job_summaries dictionary.  If this
    # uses too much memory (e.g., massive amount of input jsons), the creation
    # of job_summaries can be skipped and normalization can happen as each
    # input json is loaded.
    job_summaries = []
    for input_filename in args.file:
        input_data = []
        with open(input_filename, 'r') as input_file:
            ### if a json file can't be parsed, report which file it is
            try:
                input_data = json.load(input_file)
                # sys.stderr.write("Consumed json file %s\n" % input_filename)
            except ValueError:
                if os.path.getsize(input_filename) > 0:
                    sys.stderr.write("Malformed json file %s; skipping\n" % input_filename)
                else:
                    sys.stderr.write("Empty json file %s; skipping\n" % input_filename)
        for job_summary in input_data:
            job_summaries.append(job_summary)

    summaries_df = normalize_job_summaries(job_summaries)

    if args.output == "-":
        output_file = None
    elif args.output is None:
        output_file = "summaries_%s.csv"
    else:
        output_file = args.output

    result = save_output(summaries_df, output_file)

    if result is not None:
        print("Saved output to %s" % result)

def save_output(dataframe, output_filename):
    """
    Save the formatted output of the normalized job summaries data.  If
    output_filename is None, just return the CSV as a string.
    """
    if output_filename is not None and "%s" in output_filename:
        first, last = get_first_last_jobtimes(dataframe)
        timestamp_str = "%s-%s" % (
            first.strftime("%Y-%m-%d"),
            last.strftime("%Y-%m-%d"))
        output_filename = output_filename % timestamp_str

    result = dataframe.to_csv(path_or_buf=output_filename, index_label=False)

    if output_filename is None:
        sys.stdout.write(result)

    return output_filename

def get_first_last_jobtimes(summaries_df):
    """
    Find the earliest and latest jobs in the summaries dataframe and return
    datetime objects representing their timestamps.
    """
    earliest = long(summaries_df['_datetime_start'].min())
    latest = long(summaries_df['_datetime_start'].max())
    return datetime.datetime.fromtimestamp(earliest), \
           datetime.datetime.fromtimestamp(latest)

def normalize_job_summaries(job_summaries):
    """
    Take a list of job summary dicts and output a Pandas dataframe with the
    normalized contents of those dicts.
    """
    norm_summaries = {}
    expected_rows = 0
    for job_summary in job_summaries:
        ### add each key to a dict of lists which will become our dataframe
        cols_found = set([])
        for key, value in job_summary.items():
            cols_found.add(key)
            if key in norm_summaries:
                norm_summaries[key].append(value)
            else:
                norm_summaries[key] = [value]

        ### start looking for abnormalities in data
        if expected_rows == 0:
            expected_cols = set(job_summary.keys())
        else:
            ### if there are keys in previous sets but not this one, we need to
            ### append NaNs
            for missing_key in expected_cols - cols_found:
                norm_summaries[missing_key].append(None)
                expected_cols.add(missing_key)

            ### if there are keys that showed up for the first time in this set,
            ### we need to pad the previous rows with NaNs
            for missing_key in cols_found - expected_cols:
                save_data = norm_summaries[missing_key][0]
                norm_summaries[missing_key] = ([None] * expected_rows) + [save_data]
                expected_cols.add(missing_key)

        expected_rows += 1

    ### verify that all columns have the same number of rows
    expected_len, expected_key = -1, "_unknown"
    for key, array in norm_summaries.items():
        if expected_len < 0:
            expected_len = len(array)
            expected_key = key
        elif expected_len != len(array):
            failmsg = "%s has length %d but %s has length %d\n" % (
                expected_key,
                expected_len,
                key,
                len(array))
            raise Exception(failmsg)

    return pandas.DataFrame.from_dict(norm_summaries)

if __name__ == "__main__":
    _normalize_job_summaries()
