import numbers
import warnings
import numpy
import abcutils

def identify_contributors(dataframe, dependent_column, expected_minima=-1, want_good=False):
    """Identify secondary metrics that coincide with a good/bad primary metric

    Args:
        dataframe (DataFrame): dataframe containing one metric per column
            over a series of measurements
        dependent_column (str): name of column in `dataframe` corresponding
            to the metric to which contributors will be identified
        want_good (bool): are we identifying metrics that are unusually good
            (True) or unusually bad (False)?
        expected_minima (int): iloc of `dataframe` that is the expected local
            minima; default of -1 selects the final value in the dataframe
    Returns:
        List of dicts, where each dicts corresponds to a single metric that
        was identified as meeting the contribution criteria.  A dict
        contains the 'error' key and a True value when `dependent_column` does
        not fall within the most extreme quartile.
    """
    contributors = []
    for column in dataframe.columns:
        big_is_good = abcutils.CONFIG['metric_big_is_good'].get(column, True)

        result = None
        # we want the value to be lower than the cutoff when either
        # (a) we're looking for bad, and big IS good, or
        # (b) we're looking for good, and big IS NOT good
        if (not want_good and big_is_good) or (want_good and not big_is_good):
            try:
                cutoff = numpy.nanpercentile(dataframe[column].iloc[0:-1], 25)
            except TypeError: # if passed non-numeric columns, just skip them
                continue
            if dataframe[column].iloc[expected_minima] < cutoff:
                result = {
                    'metric': column,
                    'value': dataframe[column].iloc[expected_minima],
                    'comparator': "<",
                    'cutoff': cutoff,
                }
        # we want the value to be higher than the cutoff when either
        # (a) we're looking for good, and big IS good
        # (b) we're looking for bad, and big IS NOT good
        elif (want_good and big_is_good) or (not want_good and not big_is_good):
            try:
                cutoff = numpy.nanpercentile(dataframe[column].iloc[0:-1], 75)
            except TypeError:
                continue
            if dataframe[column].iloc[expected_minima] > cutoff:
                result = {
                    'metric': column,
                    'value': dataframe[column].iloc[-1],
                    'comparator': ">",
                    'cutoff': cutoff,
                }

        if column == dependent_column:
            if result is None and expected_minima is not None:
                warnings.warn("%s=%s (index %s) not in the %s quartile (%s) of %d values" %
                              (column,
                               dataframe[column].iloc[expected_minima],
                               dataframe[column].index[expected_minima],
                               "best" if want_good else "worst",
                               cutoff,
                               len(dataframe)))
                return [{"error": True}]
        elif result:
            contributors.append(result)

    return contributors

def count_contributors(dataframe, plot_metric, loci, min_points, want_good=False):
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

    Returns:
        Dict keyed by the columns of `dataframe` and whose values are the number
        of times each key was identified as a contributor to extreme performance.
        Also includes the following special keys:
            * `_loci_ignored`: number of loci not examined due to the window
              containing fewer than `min_points` benchmark measurements
            * `_loci_unclassified`: number of loci which had no contributors
            * `_loci_classified`: number of loci for which contributors were found
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
        expected_minima = region_df.index.get_loc(locus.Index)
        if len(region_df) < min_points:
            results['_loci_ignored'] += 1
        else:
            contributors = identify_contributors(region_df,
                                                 plot_metric,
                                                 want_good=want_good,
                                                 expected_minima=expected_minima)
            if len(contributors) == 0:
                results['_loci_unclassified'] += 1
            elif len(contributors) == 1 and contributors[0].get('error', False):
                results['_loci_errors'] += 1
            else:
                results['_loci_classified'] += 1
                for contributor in contributors:
                    results[contributor['metric']] = results.get(contributor['metric'], 0) + 1
    return results


def classify_extreme_measurements(dataframe,
                                  plot_metric,
                                  secondary_metrics,
                                  want_good,
                                  test_platforms=None,
                                  benchmark_ids=None):
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

    Returns:
        dict with the following keys:
            - totals: number of extreme values classified, unclassifiable,
                errored, or ignored
            - per_metric (dict): keyed by metrics containing total number
                of times metric is flagged
            - per_test (list of dict): each dict reports the keys flagged
                for a single extreme value.  Easy to transform into DataFrame
    """
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
                                                       maxes=want_good)
            result = count_contributors(dataframe=_filtered_df[['_datetime_start'] + secondary_metrics],
                                        plot_metric=plot_metric,
                                        loci=loci,
                                        min_points=abcutils.features.SHORT_WINDOW,
                                        want_good=want_good)
            result['_test_platform'] = _test_platform
            result['_benchmark_id'] = _benchmark_id
            results.append(result)

    results_flat = {}
    for result in results:
        for key, value in result.iteritems():
            if isinstance(value, numbers.Number):
                results_flat[key] = results_flat.get(key, 0) + value

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
