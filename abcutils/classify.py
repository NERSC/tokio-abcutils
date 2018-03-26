import numbers
import operator
import warnings
import numpy
import pandas
import abcutils

def _cutoff_percentile(dataframe, percentile):
    """Safely calculate percentile of a dataframe; return None if non-numeric
    """
    try:
        return numpy.nanpercentile(dataframe, percentile)
    except TypeError: # if passed non-numeric columns, just skip them
        return None

DEFAULT_CLASSIFIER = 'percentile'
CLASSIFIERS = {
    'percentile': {
        'compare_low': operator.lt,
        'compare_high': operator.gt,
        'cutoff_low': _cutoff_percentile,
        'cutoff_low_args': [25],
        'cutoff_high': _cutoff_percentile,
        'cutoff_high_args': [75],
    },
    'minmax': {
        'compare_low': operator.eq,
        'compare_high': operator.eq,
        'cutoff_low': numpy.min,
        'cutoff_low_args': [],
        'cutoff_high': numpy.max,
        'cutoff_high_args': [],
    }
}

def identify_contributors(region, target_column, target_index,
                          correlate_columns=None,
                          want_good=False, classifier=DEFAULT_CLASSIFIER):
    """Identify secondary metrics that coincide with a good/bad primary metric

    Args:
        region (DataFrame): region containing one metric per column
            over a series of measurements
        target_column (str): name of column in `region` corresponding
            to the metric to which contributors will be identified
        target_index: index value of `region` that is the point of interest
        correlate_columns (list of str): if specified, only consider the 
            columns specified; if None, default is to use region.columns
        want_good (bool): are we identifying metrics that are unusually good
            (True) or unusually bad (False)?
        classifier (str): name of classifier technique to use; corresponds to
            keys in abcutils.classify.CLASSIFIERS

    Returns:
        pandas.DataFrame where each row corresponds to a single metric that
        was either accepted or rejected as meeting the contribution criteria.
    """
    if classifier not in CLASSIFIERS:
        raise KeyError("invalid classifier")
    else:
        method = CLASSIFIERS[classifier]
  
    if correlate_columns is None:
        correlate_columns = region.columns

    contributors = []
    for column in correlate_columns:
        big_is_good = abcutils.CONFIG['metric_big_is_good'].get(column, True)
        series = region[column]

        # check for numericity - NaN counts as numeric, strangely, so we are
        # just discarding columns that cannot be min/max'ed
        if not isinstance(series.min(), numbers.Number):
            continue

        # we want the value to be lower than the cutoff when either
        #   (a) we're looking for bad, and big IS good, or
        #   (b) we're looking for good, and big IS NOT good
        if (not want_good and big_is_good) or (want_good and not big_is_good):
            cutoff = method['cutoff_low'](series, *(method['cutoff_low_args']))
            compare = method['compare_low']
        # we want the value to be higher than the cutoff when either
        #   (a) we're looking for good, and big IS good
        #   (b) we're looking for bad, and big IS NOT good
        elif (want_good and big_is_good) or (not want_good and not big_is_good):
            cutoff = method['cutoff_high'](series, *(method['cutoff_high_args']))
            compare = method['compare_high']
        else:
            cutoff = None
            compare = None

        if cutoff is None or numpy.isnan(cutoff):
            continue

        # initialize record
        result = {
            'metric_name': column,
            'metric_threshold': cutoff,
            'target_index': target_index,
            'target_value': series.loc[target_index],
            'target_metric_matches': False,
            'num_metric_matches': 0,
            'num_metric_observations': 0,
            'num_region_observations': len(series),
            'region_start': region.iloc[0]['_datetime_start'],
        }
        for copy_key in ['_test_platform', '_benchmark_id']:
            result[copy_key] = region.loc[target_index][copy_key]

        # if the observed value for `metric` in target_index passes the cutoff,
        # we classify it as a possible contributor
        if compare(result['target_value'], cutoff):
            result['target_metric_matches'] = True 

        # count the number of other matches
        for observation in series.values:
            if not numpy.isnan(observation):
                result['num_metric_observations'] += 1
                if compare(observation, cutoff):
                    result['num_metric_matches'] += 1

        # calculate p-value of the target_metric_matches classification as well
        # as the pvalue that a random variable would also be classified
        result['pvalue'] = numpy.float64(result['num_metric_matches']) / result['num_metric_observations']
        result['random_pvalue'] = numpy.float64(1.0) / result['num_region_observations']

        # append finding
        if column == target_column:
            if not result['target_metric_matches']:
                warnings.warn("%s=%s (index %s) not in the %s quartile (%s) of %d values" %
                              (column,
                               series.loc[target_index],
                               target_index,
                               "best" if want_good else "worst",
                               cutoff,
                               len(region)))
        else:
            contributors.append(result)

    return pandas.DataFrame.from_dict(contributors)
