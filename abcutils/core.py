"""
A set of utility functions to assist in working with TOKIO-ABC results
"""

import os
import json
import pandas

# Load system-wide constants
CONFIG = {}
with open('abcconfig.json', 'r') as config_file:
    CONFIG = json.load(config_file)

def load_and_synthesize_csv(csv_file, system="edison"):
    """
    Load a CSV file and synthesize new metrics that may be useful for subsequent
    analysis.
    """

    ### For Edison
    dataframe = pandas.read_csv(csv_file).dropna()
    dataframe['_system'] = system

    # Did job do mostly reads or mostly writes?
    dataframe['darshan_write_job?'] = [1 if x else 0 for x in dataframe['darshan_biggest_write_api_bytes'] > dataframe['darshan_biggest_read_api_bytes']]
    dataframe['darshan_read_or_write_job'] = ['write' if x == 1 else 'read' for x in dataframe['darshan_write_job?']]

    # Determine if file per process or shared-file was used predominantly.
    # If the number of files opened divides evenly by the number of processes,
    # it is file per process; otherwise, we call it shared-file.
    # "divides evenly" is defined as "evenly to within a 5% tolerance" to
    # account for one-off single-shared files like input decks, config files,
    # etc
    remainder_write = dataframe['darshan_biggest_write_api_files'] % dataframe['darshan_nprocs']
    remainder_read = dataframe['darshan_biggest_read_api_files'] % dataframe['darshan_nprocs']
    fpp_write = remainder_write / dataframe['darshan_biggest_write_api_files']
    fpp_read = remainder_read / dataframe['darshan_biggest_read_api_files']
    dataframe['darshan_fpp_write_job?'] = [1 if abs(x) < 0.05 else 0 for x in fpp_write]
    dataframe['darshan_fpp_read_job?'] = [1 if abs(x) < 0.05 else 0 for x in fpp_read]

    # Simplify the darshan_app counter
    dataframe['darshan_app'] = [os.path.basename(x) for x in dataframe['darshan_app']]

    # Calculate coverage factors
    dataframe['coverage_factor_read_bw'] = dataframe['darshan_biggest_read_fs_bytes'] / dataframe['fs_tot_bytes_read']
    dataframe['coverage_factor_write_bw'] = dataframe['darshan_biggest_write_fs_bytes'] / dataframe['fs_tot_bytes_written']
    job_nodehrs = (dataframe['darshan_nprocs'] / CONFIG['job_ppns'][system]) * dataframe['darshan_walltime'] / 3600
    dataframe['coverage_factor_nodehrs'] = job_nodehrs / dataframe['jobsdb_concurrent_nodehrs']

    # Calculate the relevant metrics for counters that have both a read and
    # writen component; mostly for convenience.
    for key in ('coverage_factor_%s_bw',
                'darshan_fpp_%s_job?',
                'darshan_biggest_%s_api_bytes'):
        new_key = key.replace('%s_', '')
        dataframe[new_key] = [dataframe.iloc[i][key % x] for i, x in enumerate(dataframe['darshan_read_or_write_job'])]
    dataframe['darshan_fpp_or_ssf_job'] = ['fpp' if x == 1 else 'shared' for x in dataframe['darshan_fpp_job?']]

    # In ABC, all shared-file I/O is performed via MPI-IO, and all
    # file-per-process is POSIX, so there is a simple 1:1 mapping.  Any
    # deviation from this in the future will require more sophisticated
    # heuristics to determine the parallel I/O API used.
    dataframe['darshan_app_api'] = ['posix' if x == 1 else 'mpiio' for x in dataframe['darshan_fpp_job?']]

    return dataframe

def normalize_column(dataframe, target_col, group_by_cols, new_col_base):
    """
    Given a dataframe, the name of a column containing raw performance
    measurements, and a list of column names in which performance should be
    normalized, create a new colum named by new_col_base with normalized
    performance.  Modifies the dataframe in-place and does not return
    anything.
    """

    norm_group = dataframe.groupby(group_by_cols)
    norm_denoms = {
        'mean': norm_group[target_col].mean(),
        'median': norm_group[target_col].median(),
        'max': norm_group[target_col].max(),
    }
    new_cols = {}

    for function, denoms in norm_denoms.iteritems():
        new_col_key = '%s_by_%s' % (new_col_base, function)
        new_cols[new_col_key] = []
        for _, row in dataframe.iterrows():
            # must descend down each rank of the grouping to get the correct
            # normalization constant
            denom = None
            for key in group_by_cols:
                if denom is None:
                    denom = denoms[row[key]]
                else:
                    denom = denom[row[key]]
            new_cols[new_col_key].append(row[target_col] / denom)

    ### Take our normalized data and add them as new columns
    for new_col, new_col_data in new_cols.iteritems():
        dataframe[new_col] = new_col_data

def apply_filters(dataframe, filter_list, verbose=False):
    """
    Applies a list of filters to a dataframe and returns the resulting view
    """
    num_rows = len(dataframe)
    if verbose:
        print "Start with %d rows before filtering" % num_rows
    net_filter = [True] * len(dataframe.index)
    for idx, condition in enumerate(filter_list):
        count = len([x for x in net_filter if x])
        net_filter &= condition
        num_drops = (count - len([x for x in net_filter if x]))
        if verbose:
            print "Dropped %d rows after filter #%d (%d left)" % (num_drops, idx, count - num_drops)

    if verbose:
        print "%d rows remaining" % len(dataframe[net_filter].index)

    return dataframe[net_filter]
