import numbers
import operator
import warnings
import numpy
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

def identify_contributors(dataframe, dependent_column, minima_iloc=-1, want_good=False, classifier=DEFAULT_CLASSIFIER):
    """Identify secondary metrics that coincide with a good/bad primary metric

    Args:
        dataframe (DataFrame): dataframe containing one metric per column
            over a series of measurements
        dependent_column (str): name of column in `dataframe` corresponding
            to the metric to which contributors will be identified
        minima_iloc (int): iloc of `dataframe` that is the expected local
            minima; default of -1 selects the final value in the dataframe
        want_good (bool): are we identifying metrics that are unusually good
            (True) or unusually bad (False)?
        classifier (str): name of classifier technique to use; corresponds to
            keys in abcutils.classify.CLASSIFIERS
    Returns:
        List of dicts, where each dicts corresponds to a single metric that
        was identified as meeting the contribution criteria.  A dict
        contains the 'error' key and a True value when `dependent_column` does
        not fall within the most extreme quartile.
    """
    if classifier not in CLASSIFIERS:
        raise KeyError("invalid classifier")
    else:
        method = CLASSIFIERS[classifier]
  
    contributors = []
    for column in dataframe.columns:
        big_is_good = abcutils.CONFIG['metric_big_is_good'].get(column, True)
        region = dataframe[column]#.iloc[0:-1]

        # we want the value to be lower than the cutoff when either
        #   (a) we're looking for bad, and big IS good, or
        #   (b) we're looking for good, and big IS NOT good
        if (not want_good and big_is_good) or (want_good and not big_is_good):
            cutoff = method['cutoff_low'](region, *(method['cutoff_low_args']))
            compare = method['compare_low']
        # we want the value to be higher than the cutoff when either
        #   (a) we're looking for good, and big IS good
        #   (b) we're looking for bad, and big IS NOT good
        elif (want_good and big_is_good) or (not want_good and not big_is_good):
            cutoff = method['cutoff_high'](region, *(method['cutoff_high_args']))
            compare = method['compare_high']
        else:
            cutoff = None
            compare = None

        if cutoff is None:
            continue

        # comparing equality when all values are identical results in spurious matches
        unique = len(region.unique())
        if compare(dataframe[column].iloc[minima_iloc], cutoff) and unique > 1:
            result = {
                'metric': column,
                'value': dataframe[column].iloc[minima_iloc],
                'cutoff': cutoff,
            }
            # calculate p-value
            num_match = 0
            num_tot = 0
            for loc, observation in dataframe[column].iteritems():
                num_tot += 1
                if compare(observation, cutoff):
                    num_match += 1
            # subtract one because we'll always match ourself in the above loop
            result['pvalue'] = float(num_match) / num_tot
        else:
            result = None

        if column == dependent_column:
            if result is None:
                warnings.warn("%s=%s (index %s) not in the %s quartile (%s) of %d values" %
                              (column,
                               dataframe[column].iloc[minima_iloc],
                               dataframe[column].index[minima_iloc],
                               "best" if want_good else "worst",
                               cutoff,
                               len(dataframe)))
                return [{"error": True}]
        elif result:
            contributors.append(result)

    return contributors

def count_contributors(dataframe, plot_metric, loci, min_points, want_good=False, classifier=DEFAULT_CLASSIFIER):
    """Count the secondary metrics that may have contributed to good/bad primary metric

    Args:
        dataframe (DataFrame): dataframe containing one metric per column
            over a series of measurements
        plot_metric (str): name of column corresponding to the primary metric
            of performance
        loci (DataFrame): output of `features.generate_loci_*` function;
            contains `region_start`, `region_end` which are indexes of
            `dataframe`, is itself indexed using the same indices as
            `dataframe`.  Each row is a locus an the region that bounded that
            locus search.
        min_points (int): the minimum number of measurements that must fall
            within each locus's region for the locus's contributors to be
            counted
        want_good (bool): are we identifying metrics that are unusually good
            (True) or unusually bad (False)?
        classifier (str): name of classifier technique to use; corresponds to
            keys in abcutils.classify.CLASSIFIERS

    Returns:
        Dict keyed by the columns of `dataframe` and whose values are the number
        of times each key was identified as a contributor to extreme performance.
        Also includes the following special keys:
            * `_loci_ignored`: number of loci not examined due to the window
              containing fewer than `min_points` benchmark measurements
            * `_loci_unclassified`: number of loci which had no contributors
            * `_loci_classified`: number of loci for which contributors were found
            * `_tot_<metric>`: number of rows that registered a non-NaN value
              for <metric>
            * `_pval_<metric>`: p-value for <metric>
    """
    results = {
        '_loci_ignored': 0,
        '_loci_unclassified': 0,
        '_loci_classified': 0,
        '_loci_errors': 0,
    }
    abcutils.CONFIG['metric_labels']['_loci_unclassified'] = "Indeterminate"
    for locus in loci.itertuples():
        region_idx0 = dataframe.index.get_loc(locus.region_start)
        region_idxf = dataframe.index.get_loc(locus.region_end)
        region_df = dataframe.iloc[region_idx0:region_idxf]
        minima_iloc = region_df.index.get_loc(locus.Index)
        if len(region_df) < min_points:
            results['_loci_ignored'] += 1
        else:
            contributors = identify_contributors(region_df,
                                                 plot_metric,
                                                 minima_iloc=minima_iloc,
                                                 want_good=want_good,
                                                 classifier=classifier)

            for metric, count in region_df.count().iteritems():
                if count > 0:
                    key = '_tot_' + metric
                    results[key] = results.get(key, 0) + 1

            if len(contributors) == 0:
                results['_loci_unclassified'] += 1
            elif len(contributors) == 1 and contributors[0].get('error', False):
                results['_loci_errors'] += 1
            else:
                results['_loci_classified'] += 1
                for contributor in contributors:
                    results[contributor['metric']] = results.get(contributor['metric'], 0) + 1
                    key = '_pval_' + contributor['metric']
                    results[key] = results.get(key, 0.0) + contributor['pvalue']

    for metric in results.keys():
        if metric.startswith('_pval_'):
            tot_key = metric.replace('_pval_', '')
            if tot_key in results:
                results[metric] /= results[tot_key]
    return results


def classify_extreme_measurements(dataframe,
                                  plot_metric,
                                  secondary_metrics,
                                  want_good,
                                  test_platforms=None,
                                  benchmark_ids=None,
                                  classifier=DEFAULT_CLASSIFIER,
                                  **kwargs):
    """Classify the sources of extreme performance

    Args:
        dataframe (pandas.DataFrame): all data from all benchmarks and file
            systems
        plot_metric (str): column name to use for locus generation
        secondary_metrics (list of str): columns which are eligible to be
            flagged as contributors to extreme values
        want_good (bool): find extremely good (True) or bad (False) values for
            plot_metric
        test_platforms (list of str): list of test platforms to subselect
            (default: all)
        benchmark_ids (list of str): list of benchmarks to subselect
            (default: all)
        classifier (str): name of classifier technique to use; corresponds to
            keys in abcutils.classify.CLASSIFIERS
        kwargs: arguments to pass to abcutils.features.generate_loci_sma
    Returns:
        dict with the following keys:
            - totals: number of extreme values classified, unclassifiable,
                errored, or ignored
            - per_metric (dict): keyed by metrics containing total number
                of times metric is flagged
            - per_test (list of dict): each dict reports the keys flagged
                for a single extreme value.  Easy to transform into DataFrame
    """
    args = {
        'short_window': abcutils.features.SHORT_WINDOW,
        'long_window': abcutils.features.LONG_WINDOW,
        'min_width': abcutils.features.MIN_REGION,
    }
    args.update(kwargs)

    if not test_platforms:
        test_platforms = sorted(dataframe['_test_platform'].unique())
    if not benchmark_ids:
        benchmark_ids = sorted(dataframe['_benchmark_id'].unique())

    grouped_df = dataframe.groupby(by=['_test_platform', '_benchmark_id'])

    results = []
    for _test_platform in test_platforms:
        for _benchmark_id in benchmark_ids:
            try:
                _filtered_df = grouped_df.get_group((_test_platform, _benchmark_id))
            except KeyError:
                continue
            loci = abcutils.features.generate_loci_sma(_filtered_df,
                                                       plot_metric,
                                                       mins=(not want_good),
                                                       maxes=want_good,
                                                       **args)
            result = count_contributors(dataframe=_filtered_df[['_datetime_start'] + secondary_metrics],
                                        plot_metric=plot_metric,
                                        loci=loci,
                                        min_points=abcutils.features.MIN_REGION,
                                        want_good=want_good,
                                        classifier=classifier)
            result['_test_platform'] = _test_platform
            result['_benchmark_id'] = _benchmark_id
            results.append(result)

    results_flat = {}
    count_flat = {}
    for result in results:
        for key, value in result.iteritems():
            if isinstance(value, numbers.Number):
                results_flat[key] = results_flat.get(key, 0) + value
                count_flat[key] = count_flat.get(key, 0) + 1

    drop_keys = []
    for key in results_flat:
        if key.startswith('_pval_'):
            base_key = key.replace('_pval_', '')
            if base_key in count_flat:
                results_flat[key] /= count_flat[key]
            else:
                # throw out pvalues we can't average
                drop_keys.append(key)
    for key in drop_keys:
        del results_flat[key]

    counts = {
        'classified': results_flat.pop('_loci_classified'),
        'unclassified': results_flat.get('_loci_unclassified'),
        'ignored': results_flat.pop('_loci_ignored'),
        'errors': results_flat.pop('_loci_errors'),
    }
    counts['total'] = counts['classified'] \
                      + counts['unclassified'] \
                      + counts['ignored'] \
                      + counts['errors']
    return {
        'totals': counts,
        'per_metric': results_flat,
        'per_test': results,
    }
