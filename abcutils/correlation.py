"""
Routines for calculating correlation metrics from TOKIO-ABC data
"""

import pandas
import numpy
import scipy.stats
import abcutils.core as core

def calc_correlation_vector(dataframe, correlate_with, analysis_func=scipy.stats.pearsonr,
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

def autocorrelation(dataset, loci, xmin, xmax, delta, norm_by_locus=False, norm=False,
                    agg_func=scipy.stats.gmean, stdev_func=core.geometric_stdev):
    """Calculate the autocorrelation function for a dataset
    
    Calculate the autocorrelation of an arbitrary metric.
    
    Args:
        dataset (list of tuples): list of (x, y) tuples over which the PDF
            should be calculated
        loci (list): x values of interest around which the PDF should be calculated
        xmin: minimum value of x in the resulting PDF
        xmax: maximum value of x in the resulting PDF
        delta: resolution of PDF function expressed in the same units of x
        norm_by_locus (bool): express PDF in terms of fraction performance
            relative to each locus
        norm (bool): express PDF in terms of fraction performance relative to
            global mean
        agg_func (function): function to apply to each list of data points to
            generate y values
        stdev_func (function): function to calculate the standard deviation of
            each list of data points
    Returns:
        (xbins, ybins, nbins) where
        xbins are the x values of the pair distribution function
        ybins are the y values of the pair distribution function
        nbins are the number of y values that fell into each bin
    """
    width = xmax - xmin
    num_bins = long(width / delta)
    xbins = numpy.arange(num_bins, dtype='float64') * delta + xmin
    ybin_vals = [[] for _ in range(num_bins)] # list of lists containing all raw values of each bin

    # convert to dataframe so we can do some fancy indexing
    dataset_df = pandas.DataFrame(dataset, columns=['x', 'y']).set_index('x')
    for locus in loci:
        locus_y = dataset_df.loc[locus][0]
        for row in dataset_df.itertuples():
            x, y = row[0], row[1]
            dx = x - locus
            
            if abs(dx) > width:
                continue

            # how many bins away from zero - we round up when necessary to
            # avoid jobs from the first day finishing a few minutes under
            # 24 hours and thereby falling in the 0th day's bin
            x_bin = long(round(dx / delta))
            x_bin -= long(round(xbins[0] / delta))
            
            # we drop everything in the 0th bin (e.g., if multiple jobs ran
            # on the same day) because they cause artifacts in autocorrelation
            if x_bin == 0 and x != locus:
                continue

            # normalize signal by 
            if norm_by_locus:
                y_val = y / locus_y
            else:
                y_val = y

            if x_bin < xbins.shape[0] and x_bin >= 0:
                ybin_vals[x_bin].append(y_val)

    # Convert bins of measurements into aggregate values at each x bin
    ybins = numpy.nan_to_num(numpy.array([agg_func(x) for x in ybin_vals]))
    nbins = numpy.array([len(x) for x in ybin_vals])
    # standard deviation
    sbins = numpy.nan_to_num(numpy.array([stdev_func(x) for x in ybin_vals]))
    # first and third quartiles
    qbins = numpy.array([[numpy.percentile(x, 25) if x else 0.0 for x in ybin_vals],
                         [numpy.percentile(x, 75) if x else 0.0 for x in ybin_vals]])

    # normalize to the global average measurement?
    if norm:
        global_mean = agg_func(dataset_df['y'])
        ybins /= global_mean
        sbins /= global_mean
        qbins /= global_mean
    
    return xbins, ybins, nbins, qbins
