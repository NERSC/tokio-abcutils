import os
import datetime
import numpy
import pandas
import abcutils.core

DATE_START = datetime.datetime(2017, 2, 14)
DATE_END = datetime.datetime(2018, 2, 15)
INPUT_DATASETS = {
    'edison': 'summaries/edison-summaries_2017-02-14-2018-02-15.csv.gz',
    'cori': 'summaries/cori-summaries_2017-02-14-2018-02-15.csv.gz',
    'mira': 'summaries/mira-summaries_2017-02-14_2018-02-15.csv.gz',
}
CACHE_FILE = 'cache.hdf5'

def load_raw_datasets(input_datasets=None, cache_file=CACHE_FILE, verbose=True):
    """Load data from CSVs and synthesize metrics

    Args:
        input_datasets (dict): keyed by system name (mira, edison, cori); values
            are path to the CSV containing the data for that system.
        cache_file (str): path to a cache file to load.  If does not exist,
            create it after the CSVs are loaded
        verbose (bool): announce what is happening before it happens
    Returns:
        Concatenated pandas.DataFrame with data from all CSVs
    """
    if input_datasets is None:
        input_datasets = INPUT_DATASETS

    # Load raw datasets; use cached version if available
    if cache_file and os.path.isfile(cache_file):
        if verbose:
            print("Loading from cache %s" % cache_file)
        dataframe = pandas.read_hdf(cache_file, 'summary')
    else:
        dataframes = []
        for system, csvfile in input_datasets.items():
            dataframes.append(abcutils.load_and_synthesize_csv(csvfile, system=system))
        dataframe = pandas.concat(dataframes, axis='rows')
        if cache_file:
            if verbose:
                print("Cached synthesized CSV to %s" % cache_file)
            dataframe.to_hdf(cache_file,
                             key='summary',
                             mode='w',
                             format='fixed',
                             complevel=9,
                             complib='zlib')

    return dataframe

def build_sc18_filters(dataframe):
    """Build generic data filters for the SC paper

    Returns:
        dataframe (pandas.DataFrame): Raw dataset from load_and_synthesize_csv

    Returns:
        list: List of filters to be passed to ``abcutils.core.apply_filters``
        along with ``dataframe``
    """
    filters = []

    # Constrain dates to those covered by the paper
    filters.append(dataframe['_datetime_start'] < DATE_END)
    filters.append(dataframe['_datetime_start'] >= DATE_START)

    # Drop Darshan logs from jobs that didn't actually do significant I/O; this
    # filters out a set of VPIC jobs that hit a bug (related to the Edison
    # upgrade) that allowed them to finish correctly but never write their data
    # out.
    filters.append(dataframe['darshan_total_gibs_posix'] > 1.0)

    # Some of the Mira data has invalid benchmark_ids; drop them
    filters.append(dataframe['_benchmark_id'] != 'hacc_io_write_shared_write')

    # The Haswell data is misleading since it used a tiny fraction of the system
    filters.append(dataframe['_test_platform'] != 'cscratch@cori-haswell')

    return filters

def load_dataset(verbose=True, truncate_contention=False, drop_cf_above=1.2, filter_func=build_sc18_filters, *args, **kwargs):
    """Load dataset used for Year in the Life paper

    Load the canonical dataset used for the "Year in the Life" paper, apply
    global filters on the dataset, and add a few additional derived metrics.

    Args:
        verbose (bool): Print messages describing from where data is being
            loaded
        truncate_contention (bool): If True, apply max(0.0, val) to all
            derived contention values.  Default value corresponds to what
            was used in the paper.
        drop_cf_above (float or None): Drop any records whose coverage factors
            for bandwidth are above this value.  Default value corresponds to
            what was used in the paper.
        filter_func: Function that takes a dataframe as an argument and returns
            a list of filters that can be passed to
            ``abcutils.core.apply_filters()``

    Returns:
        pandas.DataFrame: Loaded, filtered, and augmented dataset
    """
    dataframe = load_raw_datasets(verbose=verbose, *args, **kwargs)

    # Reset the index to ensure that there are no degenerate indices in the final dataframe
    dataframe.index = pandas.Index(data=numpy.arange(len(dataframe)), dtype='int64')

    # Apply a filter to invalidate obviously bogus bandwidth coverage factors
    if drop_cf_above is not None:
        for index in dataframe[dataframe['coverage_factor_bw'] > drop_cf_above].index:
            dataframe.loc[index, 'coverage_factor_bw'] = numpy.nan

    # Drop some of the weird columns left over from the CSV
    dataframe = dataframe.drop(
        columns=[x for x in ['Unnamed: 0', 'index'] if x in dataframe.columns],
        axis=1)

    if filter_func:
        filtered_df = abcutils.core.apply_filters(dataframe, filter_func(dataframe), verbose).sort_values('_datetime_start').copy()
    else:
        filtered_df = dataframe.sort_values('_datetime_start').copy()

    # Reset the index to ensure that there are no degenerate indices in the final dataframe
    filtered_df.index = pandas.Index(data=numpy.arange(len(filtered_df)), dtype='int64')

    del dataframe

    return filtered_df
