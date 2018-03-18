"""
A set of utility functions to assist in working with TOKIO-ABC results
"""

import os
import gzip
import time
import datetime
import mimetypes
import pandas
import numpy
import scipy.stats
import abcutils

def load_and_synthesize_csv(csv_file, system="edison"):
    """
    Load a CSV file and synthesize new metrics that may be useful for subsequent
    analysis.
    """

    _, encoding = mimetypes.guess_type(csv_file)
    if encoding == 'gzip':
        filep = gzip.open(csv_file, 'r')
    else:
        filep = open(csv_file, 'r')
    dataframe = pandas.read_csv(csv_file).dropna()
    filep.close()

    dataframe['_system'] = system
    def classify_subsystem(concat):
        """Distinguish cori-knl jobs from cori-haswell jobs

        Args:
            concat (str): string of form "_system darshan_nprocs"

        Returns:
            String which is either the first space-delimited token in `concat`,
            'cori-knl', or 'cori-haswell'
        """
        system, nprocs = concat.split(None, 2)
        if system == "cori":
            if int(nprocs) > 1024:
                return 'cori-knl'
            return 'cori-haswell'
        return system

    dataframe['_subsystem'] = dataframe[['_system', 'darshan_nprocs']]\
        .apply(lambda x: "%s %d" % (x[0], x[1]), axis=1)\
        .apply(classify_subsystem)

    dataframe['_test_platform'] = dataframe['_file_system'] + '@' + dataframe['_subsystem']

    # Convert timestamps to datetime objects.  Try both epoch timestamps and datetime strings.
    for datetime_field in '_datetime_start', '_datetime_end':
        if isinstance(dataframe[datetime_field].iloc[0], basestring):
            dataframe[datetime_field] = pandas.to_datetime(dataframe[datetime_field])
        else:
            dataframe[datetime_field] = dataframe[datetime_field].apply(lambda x: datetime.datetime.fromtimestamp(x))

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
    dataframe['darshan_agg_perf_by_slowest_posix_gibs'] = dataframe['darshan_agg_perf_by_slowest_posix'] / 1024.0

    # Simplify the darshan_app counter
    dataframe['darshan_app'] = [os.path.basename(x) for x in dataframe['darshan_app']]

    # Calculate coverage factors
    dataframe['coverage_factor_read_bw'] = (dataframe['darshan_biggest_read_fs_bytes'] / dataframe['fs_tot_bytes_read']).replace([numpy.inf, -numpy.inf], numpy.nan)
    dataframe['coverage_factor_write_bw'] = (dataframe['darshan_biggest_write_fs_bytes'] / dataframe['fs_tot_bytes_written']).replace([numpy.inf, -numpy.inf], numpy.nan)
    job_nodehrs = (dataframe['darshan_nprocs'] / abcutils.CONFIG['job_ppns'][system]) * dataframe['darshan_walltime'] / 3600
    if 'jobsdb_concurrent_nodehrs' in dataframe.columns:
        dataframe['coverage_factor_nodehrs'] = (job_nodehrs / dataframe['jobsdb_concurrent_nodehrs']).replace([numpy.inf, -numpy.inf], numpy.nan)
    dataframe['fs_tot_bytes'] = dataframe['fs_tot_bytes_read'] + dataframe['fs_tot_bytes_written']
    dataframe['coverage_factor_bw'] = dataframe['darshan_total_gibs_posix'] / dataframe['fs_tot_bytes'] * 2.0**30

    # Calculate the relevant metrics for counters that have both a read and
    # writen component; mostly for convenience.
    for key in ('darshan_fpp_%s_job?',
                'darshan_biggest_%s_api_bytes'):
        new_key = key.replace('%s_', '')
        dataframe[new_key] = [dataframe.iloc[i][key % x] for i, x in enumerate(dataframe['darshan_read_or_write_job'])]
    dataframe['darshan_fpp_or_ssf_job'] = ['fpp' if x == 1 else 'shared' for x in dataframe['darshan_fpp_job?']]

    # In ABC, all shared-file I/O is performed via MPI-IO, and all
    # file-per-process is POSIX, so there is a simple 1:1 mapping.  Any
    # deviation from this in the future will require more sophisticated
    # heuristics to determine the parallel I/O API used.
    dataframe['darshan_app_api'] = ['posix' if x == 1 else 'mpiio' for x in dataframe['darshan_fpp_job?']]

    # Aggregate some metadata ops
    if 'fs_tot_openclose_ops' not in dataframe.columns:
        dataframe['fs_tot_openclose_ops'] = dataframe['fs_tot_open_ops'] + dataframe['fs_tot_close_ops']

    if 'fs_tot_metadata_ops' not in dataframe.columns:
        metadata_ops_cols = [x for x in dataframe.columns if (x.startswith('fs_tot') and x.endswith('_ops'))]
        dataframe['fs_tot_metadata_ops'] = dataframe[metadata_ops_cols].sum(axis=1)

    # Calculate a benchmark id for ease of aggregation
    dataframe['_benchmark_id'] = dataframe['darshan_app'] + "_" \
        + dataframe['darshan_fpp_or_ssf_job'] + "_" \
        + dataframe['darshan_read_or_write_job']

    # Calculate normalized performance metrics (modifies data in-place)
    normalize_column(
        dataframe=dataframe,
        target_col='darshan_agg_perf_by_slowest_posix',
        group_by_cols=['darshan_app', '_subsystem', '_file_system', 'darshan_fpp_or_ssf_job', 'darshan_read_or_write_job'],
        new_col_base='darshan_normalized_perf')

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

def geometric_stdev(vector):
    """Calculate the geometric standard deviation of a vector
    Args:
        vector: any iterable of numerics
    Returns:
        the geometric standard deviation of `vector`
    """
    return numpy.exp(numpy.sqrt(
        sum([(numpy.log(xi / scipy.stats.gmean(vector)))**2.0 for xi in vector])
        / (len(vector) - 1)))

def pd2epoch(timestamp):
    """Convert a pandas.Timestamp to seconds-since-epoch

    Args:
        timestamp (pandas.Timestamp): value to convert from
    Returns:
        float representing seconds since epoch in UTC
    """
    return time.mktime(timestamp.to_pydatetime().timetuple())
