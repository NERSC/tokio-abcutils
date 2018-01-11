"""
Routines for calculating correlation metrics from TOKIO-ABC data
"""

import pandas
import scipy.stats

def calc_correlations(dataframe, correlate_with, analysis_func=scipy.stats.pearsonr,
                      ignore_cols=None):
    """
    Calculate correlation coefficients and their associated p-values for various
    counter pairs.

    analysis_func = scipy.stats.pearsonr or scipy.stats.spearmanr

    correlate_with = if None, return every possible combination of columns in
    dataframe is attempted; otherwise, the column identified by correlate_with
    is compared against all other columns

    ignore_cols = list of column headers to not correlate against
    """
    results = {}
    if ignore_cols is None:
        ignore_cols = []
    for column in dataframe.columns:
        if column == correlate_with or column in ignore_cols:
            continue
        if dataframe[column].nunique() < 2: # cannot correlate if all signals are identical
            continue
        try:
            # The Scipy stats package barfs if x or y contain any
            # NANs, but we don't want to drop all records that
            # contain any nans.  So, we wait until the very last
            # minute to drop only those columns that contain nans
            # that we would otherwise try to correlate.
            correlation = dataframe[[correlate_with, column]].dropna()
            coeff, pval = analysis_func(correlation[correlate_with], correlation[column])
        except TypeError: # non-numeric column
            continue
        results[column] = { 'coefficient': coeff, 'p-value': pval }

    return pandas.DataFrame.from_dict(results, orient='index').sort_values(by=['p-value'])
