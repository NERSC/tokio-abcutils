import time
import warnings
import pandas
import scipy.stats
import abcutils

SHORT_WINDOW = pandas.Timedelta(days=7)
LONG_WINDOW = pandas.Timedelta(days=28)
# minimum number of data points in a valid region
MIN_REGION = 7

class Streak(object):
    """
    Simple object to track the state of a streak of monotonic values
    """
    def __init__(self, start, start_val, min_streak=3):
        """
        Instantiate a Streak object
        
        Args:
            start (int): an integer index corresponding to the start of a streak
            start_val: an arbitrary numeric value corresponding to the start of
                a streak
            min_streak (int): the minimum number of values in a streak before it
                is considered a valid streak
        """
        self.indices = [start]
        self.values = [start_val]

        self.end = None
        self.dir = None
        self.min_streak = min_streak

    def update(self, index, value):
        """
        Attempt to add a new value to a monotonic stream
        
        Args:
            index (int): an index (x value) corresponding to the value being
                inserted
            value: an arbitrary numeric (y value) corresponding to index
        Returns:
            True if index, value corresponds to the direction of the streak;
                false if it is not monotonic (and therefore was not appended to
                this object)
        """
        if (self.dir == 'up' and value >= self.values[-1]) \
        or (self.dir == "down" and value <= self.values[-1]) \
        or (self.dir is None):
            self.indices.append(index)
            self.values.append(value)
            if self.dir is None and len(self.indices) > 1:
                self.dir = 'up' if self.values[-1] > self.values[-2] else 'down'
            return True
        else:
            return False # end of streak
        
    def report(self):
        """
        Return a dictionary of summary values representing this streak
        """
        if len(self.values) >= self.min_streak:
            return {
                'start': self.values[0],
                'end': self.values[-1],
                'start_index': self.indices[0],
                'end_index': self.indices[-1],
                'delta': self.values[-1] - self.values[0],
                'length': len(self.indices),
            }

    def print_report(self):
        """
        Print the summary values representing this streak in a concise and
        vaguely human-readable fashion
        """
        report = self.report()
        if report is not None:
            print "%(start).3f@%(start_index)d to %(end).3f@%(end_index)d (%(delta).3f over %(length)d steps)" % report

def find_streaks_df(dataframe, column, min_streak=3):
    """
    Find a streak of monotonically increasing or decreasing values in a
    DataFrame.  Returns DataFrame index values which can be applied to other
    views of the same DataFrame.

    Args:
        dataframe (DataFrame): any dataframe to examine
        column (str): name of the column to look for streaks
        min_streak (int): The minimum number of consecutively monotonic values
            before a streak is reported
    """
    iterable = dataframe[column]
    result = []
    for streak in find_streaks(iterable, min_streak): 
        new_indices = [iterable.index[x] for x in streak[0]]
        result.append((new_indices, streak[1]))
    return result

def find_streaks(iterable, min_streak=3):
    """
    Find a streak of monotonically increasing or decreasing values

    Args:
        iterable: any enumerable object
        min_streak (int): The minimum number of consecutively monotonic values
            before a streak is reported

    Returns:
        List of streak tuples, where each tuple[0] is a list of indices of
            `iterable` and each tuple[1] is a list of values from `iterable` for
            the streak.  The length of both elements of this tuple will be the
            same.
    """
    streak = None
    result = []
    for index, value in enumerate(iterable):        
        if streak is None:
            streak = Streak(index, value, min_streak=min_streak)
        else:
            success = streak.update(index, value)
            if not success:
                if streak.report():
                    result.append((streak.indices, streak.values))
                last_index = streak.indices[-1]
                last_value = streak.values[-1]
                streak = Streak(last_index, last_value, min_streak=min_streak)
                success = streak.update(index, value)
                assert success
    return result

def sliding_window_slopes(dataframe, column, start, end, width, delta):
    """
    Calculate the slope for all measurements within a sliding window of time.

    Args:
        dataframe (pandas.DataFrame): dataframe from which sliding window slopes
            should be calculated
        column (str): name of column over in dataframe from which sliding-window
            slopes should be calculated 
        start (datetime.datetime): trailing edge of first window
        end (datetime.datetime): leading edge of last window
        width (datetime.timedelta): width of window over which slope will be calculated
        delta (datetime.timedelta): increment by which window should be slid
            between slope calculations

    Returns:
        List of streak tuples, where each tuple[0] is a list of
            datetime.datetime objects (x values) and each tuple[1] is a list of
            interpolated y values.  The length of both elements of this tuple
            will be the same and are governed by `width` / `delta`.
    """

    result = []
    date = start
    while date + width < end:
        date_filter = dataframe['_datetime_start'] >= date
        date_filter &= dataframe['_datetime_start'] < date + width 
        subset_df = dataframe[date_filter]
        
        # x has units of seconds
        x = subset_df['_datetime_start'].apply(lambda x: time.mktime(x.to_pydatetime().timetuple()))
 
        # y has units governed by `column`
        y = subset_df[column]
       
        if len(y) > 0:
            regression = scipy.stats.linregress(x=x, y=y)
            result.append((x.values, x.apply(lambda xx: regression.slope * xx + regression.intercept).values))
        
        date += delta 

    return result

def sma_by_row(dataframe, x_column, y_column, window, **kwargs):
    """Calculate the simple moving average for a column of a dataset

    Calculates the SMA using a window that is expressed in number of rows rather
    than the units of measurement contained in those rows.

    Args:
        dataframe (pandas.DataFrame): dataframe from which sliding window slopes
            should be calculated
        x_column (str): name of column to treat as x values when calculating
            the simple moving average
        y_column (str): name of column over which simple moving average should
            be calculated
        window (int): number of consecutive dataframe rows to include in the
            simple moving average
        kwargs: parameters to pass to pandas.Series.rolling

    Returns:
        Series of simple moving averages indexed by x_column
    """
    default_args = {
        'min_periods': 1,
        'window': window,
        'center': True,
    }
    default_args.update(kwargs)

    filtered_series = dataframe[[x_column, y_column]].set_index(x_column).iloc[:, 0]

    return filtered_series.rolling(window=window, **kwargs).mean().sort_index()

def sma_by_value(dataframe, x_column, y_column, window, **kwargs):
    """Calculate the simple moving average for a column of a dataset

    Calculates the SMA using a window that is expressed in the units of
    measurement contained in those rows rather than the number of rows.

    Args:
        dataframe (pandas.DataFrame): dataframe from which sliding window slopes
            should be calculated
        x_column (str): name of column to treat as x values when calculating
            the simple moving average
        y_column (str): name of column over which simple moving average should
            be calculated
        window: width of the window, expressed in the units of y_column, over
            which the SMA will be calculated

    Returns:
        Series of simple moving averages indexed by x_column
    """

    half_window = window / 2.0
    indices = []
    values = []
    for index, start, _ in dataframe[[x_column, y_column]].itertuples():
        window_start = start - half_window
        window_end = start + half_window
        window_df = dataframe[dataframe[x_column] >= window_start]
        window_df = window_df[window_df[x_column] < window_end]
        indices.append(window_df.loc[index][x_column])
        values.append(window_df[y_column].mean())

    return pandas.Series(values, index=indices, name=y_column).sort_index()

def calculate_sma(dataframe, x_column, y_column, window, method='value', **kwargs):
    """Calculate the simple moving average for a column of a dataset

    Dispatches an SMA calculation function

    Args:
        dataframe (pandas.DataFrame): dataframe from which sliding window slopes
            should be calculated
        x_column (str): name of column to treat as x values when calculating
            the simple moving average
        y_column (str): name of column over which simple moving average should
            be calculated
        window: width of the window over which the SMA will be calculated
        method (str): 'value' or 'row'
        kwargs: parameters to pass to SMA calculator

    Returns:
        Series of simple moving averages indexed by x_column
    """
    methods = {
        'value': sma_by_value,
        'row': sma_by_row,
    }
    method = methods.get(method)
    if not method:
        raise KeyError("Invalid method (%s)" % ', '.join(methods.keys()))

    return method(dataframe=dataframe, x_column=x_column, y_column=y_column, window=window, **kwargs)

def sma_intercepts(dataframe, column, short_window, long_window, min_width=None, **kwargs):
    """Identify places where two simple moving averages intercept

    Args:
        dataframe (pandas.DataFrame): dataframe from which sliding window slopes
            should be calculated
        column (str): name of column over in dataframe from which sliding-window
            slopes should be calculated 
        short_window (int): number of consecutive dataframe rows to include in
            the short window
        long_window (int): number of consecutive dataframe rows to include in
            the long window
        min_width: minimum width, expressed in units of `x_column`, below which
            an intercept should be disregarded as a valid end of a window
        kwargs: arguments to be passed to calculate_sma()

    Returns:
        DataFrame with indices corresponding to dataframe
    """
    x_column = '_datetime_start'

    sma_short = calculate_sma(dataframe, x_column, column, window=short_window, **kwargs)
    sma_long = calculate_sma(dataframe, x_column, column, window=long_window, **kwargs)

    results = {
        x_column: [],
        'positive': [],
        'sma_short': [],
        'sma_long': [],
    }

    # Walk through both SMAs and find intercepts
    above = None
    for index, value in enumerate(sma_short):
        if above is None:
            above = value > sma_long[index]
        above_now = value > sma_long[index]
        if above_now != above:
            results[x_column].append(sma_short.index[index])
            results['positive'].append(not above)
            results['sma_short'].append(value)
            results['sma_long'].append(sma_long[index])
        above = above_now

    # Now convert sma index values to dataframe.index values
    x_series = dataframe[x_column]
    results['index'] = [x_series[x_series == x].index[0] for x in results[x_column]]

    # Remove regions that are below min_width
    result_df = pandas.DataFrame(results).set_index('index') 
    keep = [True] * len(result_df)
    if min_width is not None:
        for iloc in range(len(result_df)):
            if iloc > 0:
                region_width = result_df.iloc[iloc][x_column] - result_df.iloc[iloc - 1][x_column]
                keep[iloc] = (region_width >= min_width)

    return result_df[keep]

def sma_centroids(dataframe, column, short_window, long_window, min_width=None, **kwargs):
    """Identify centermost point between two SMA interception points

    Define regions as being bounded by two consecutive interceptions of SMAs
    with different window widths, then choose the centermost data point within
    that region.  Useful for defining regions that capture the crossover of
    SMAs.

    Args:
        dataframe (pandas.DataFrame): dataframe from which sliding window slopes
            should be calculated
        column (str): name of column over in dataframe from which sliding-window
            slopes should be calculated 
        short_window (int): number of consecutive dataframe rows to include in
            the short window
        long_window (int): number of consecutive dataframe rows to include in
            the long window
        min_width: minimum width, expressed in units of `x_column`, below which
            an intercept should be disregarded as a valid end of a window
        kwargs: arguments to be passed to calculate_sma()

    Returns:
        DataFrame with indices corresponding to dataframe
    """
    x_column = '_datetime_start'

    intercepts = sma_intercepts(dataframe, column, short_window, long_window, min_width=min_width, **kwargs)
    sma_short = calculate_sma(dataframe, x_column, column, window=short_window, **kwargs)
    sma_long = calculate_sma(dataframe, x_column, column, window=long_window, **kwargs)

    results = {
        x_column: [],
        'positive': [],
        'sma_short': [],
        'sma_long': [],
    }

    prev_index = None
    for index in intercepts.index:
        if prev_index is not None:
            region_idx0 = dataframe.index.get_loc(prev_index)
            region_idxf = dataframe.index.get_loc(index)
            region = dataframe.iloc[region_idx0:region_idxf][x_column]
            if len(region) < 3: # can find a centroid without three points
                continue
            halfway_val = region.iloc[0] + (region.iloc[-1] - region.iloc[0]) / 2.0
            closest_val = (dataframe[x_column] - halfway_val).abs()
            closest_index = closest_val[closest_val == closest_val.min()].index[0]
            closest_iloc = dataframe.index.get_loc(closest_index)
            results[x_column].append(dataframe.loc[closest_index][x_column])
            results['positive'].append(intercepts.loc[index]['positive'])
            results['sma_short'].append(sma_short.iloc[closest_iloc])
            results['sma_long'].append(sma_long.iloc[closest_iloc])
        prev_index = index

    # Now convert sma index values to dataframe.index values
    x_series = dataframe[x_column]
    results['index'] = [x_series[x_series == x].index[0] for x in results[x_column]]

    return pandas.DataFrame(results).set_index('index') 

def sma_local_minmax(dataframe, column, short_window, long_window, min_domain=3,
                     min_func=pandas.Series.idxmin,
                     max_func=pandas.Series.idxmax):
    """Identify local minima and maxima for non-overlapping SMA regions

    Segment `dataframe` into regions based on where their short and long SMAs
    intersect, and then calculate the minimum (if short < long) or maximum (if
    short > long) value within each region.

    Args:
        dataframe (pandas.DataFrame): dataframe from which sliding window slopes
            should be calculated
        column (str): name of column over in dataframe from which sliding-window
            slopes should be calculated 
        short_window (int): number of consecutive dataframe rows to include in
            the short window
        long_window (int): number of consecutive dataframe rows to include in
            the long window
        min_domain (int): ignore local minima calculated from sets of rows with
            fewer than `min_domain` rows
        min_func (function): function to apply to a Series describing a region
            to determine the index corresponding to the most extreme value for
            regions where the short SMA < long SMA
        max_func (function): function to apply to a Series describing a region
            to determine the index corresponding to the most extreme value for
            regions where the short SMA > long SMA

    Returns:
        DataFrame with indices corresponding to dataframe
    """
    x_column = '_datetime_start'

    intercepts = sma_intercepts(dataframe, column, short_window, long_window)

    results = {
        'index': [],
        x_column: [],
        'positive': [],
        column: [],
        'region_start': [],
        'region_end': [],
    }

    # Iterate through all the transition points and identify mins and maxes
    # between each intersection.  Note that we are ignoring the unbounded
    # first and last regions
    prev_row = None
    for row in intercepts.itertuples():
        if prev_row is not None:
            minmax_idx = None
            region_idx0 = dataframe.index.get_loc(prev_row.Index)
            region_idxf = dataframe.index.get_loc(row.Index)
            region = dataframe.iloc[region_idx0:region_idxf][column]
            if len(region) >= min_domain:
                if prev_row.positive and not row.positive:
                    minmax_idx = max_func(region)
                elif not prev_row.positive and row.positive:
                    minmax_idx = min_func(region)
                if minmax_idx:
                    results['index'].append(minmax_idx)
                    results[x_column].append(dataframe.loc[minmax_idx][x_column])
                    results['positive'].append(prev_row.positive)
                    results[column].append(dataframe.loc[minmax_idx][column])
                    results['region_start'].append(prev_row.Index)
                    results['region_end'].append(row.Index)
        prev_row = row

    return pandas.DataFrame(results).set_index('index')

def intercepts_to_region(dataframe, intercepts, min_domain=None):
    """Generate regions from a dataframe and sma_intercepts

    Args:
        dataframe (pandas.DataFrame): dataframe from which regions should be
            generated
        intercepts (pandas.DataFrame or pandas.Series): object with same index
            as `dataframe` whose index values denote intercepts.  Typically the
            output of `sma_intercepts()` is used here.
        min_domain (int): ignore local minima calculated from sets of rows with
            fewer than `min_domain` rows
    """
    prev_index = None
    for index in intercepts.index:
        if prev_index is not None:
            region_idx0 = dataframe.index.get_loc(prev_index)
            region_idxf = dataframe.index.get_loc(index)
            region = dataframe.iloc[region_idx0:region_idxf]
            if (not min_domain) or (len(region) >= min_domain):
                yield region
        prev_index = index

def generate_loci_sma(dataframe, column, mins, maxes, **kwargs):
    """Identify min and max values using simple moving average

    Args:
        dataframe (pandas.DataFrame): dataframe from which loci should be
            identified
        column (str): name of column over in dataframe from which sliding-window
            slopes should be calculated 
        mins (bool): True if minima should be identified in each region
        maxes (bool): True if maxima should be identified in each region
        kwargs: arguments to be passed to abcutils.features.sma_local_minmax()

    Returns:
        pandas.DataFrame indexed as `dataframe` with columns
         * `_datetime_start`, Timestamp values corresponding to each locus
         * `_region_start`, index values corresponding to the lower bound of each locus's region
         * `_region_end`, index values corresponding to the upper bound of each locus's region
    """
    args = {
        'short_window': SHORT_WINDOW,
        'long_window': LONG_WINDOW,
        'min_domain': MIN_REGION,
    }
    args.update(kwargs)

    min_vals = None
    max_vals = None
    if mins:
        min_vals = abcutils.features.sma_local_minmax(
            dataframe=dataframe,
            column=column,
            max_func=pandas.Series.idxmin,
            **args)
    if maxes:
        max_vals = abcutils.features.sma_local_minmax(
            dataframe=dataframe,
            column=column,
            min_func=pandas.Series.idxmax,
            **args)
    return pandas.concat((min_vals, max_vals)).sort_values('_datetime_start')

def generate_loci_peakdetect(dataframe, column, mins, maxes, **kwargs):
    """Identify min and max values using peakdetect() method

    Args:
        dataframe (pandas.DataFrame): dataframe from which loci should be
            identified
        column (str): name of column over in dataframe from which sliding-window
            slopes should be calculated 
        mins (bool): True if minima should be identified in each region
        maxes (bool): True if maxima should be identified in each region
        kwargs: arguments to be passed to abcutils.features.sma_local_minmax()


    Returns:
        pandas.Series indexed as `dataframe`, named `_datetime_start`,
        and containing `_datetime_start` values corresponding to loci.
    """
    args = {
        'lookahead': 7
    }
    args.update(kwargs)

    highs, lows = peakdetect.peakdetect(dataframe[column].sort_index(ascending=False),
                                                 dataframe.sort_index(ascending=False).index,
                                                 **args)
    min_vals = None
    max_vals = None
    if mins:
        indices = [x[0] for x in lows]
        min_vals = dataframe.loc[indices]['_datetime_start']
    if maxes:
        indices = [x[0] for x in highs]
        max_vals = dataframe.loc[indices]['_datetime_start']
    return pandas.concat((min_vals, max_vals)).sort_values()
