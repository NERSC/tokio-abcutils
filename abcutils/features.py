import time
import warnings
import pandas
import scipy.stats

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

def calculate_sma(dataframe, x_column, y_column, window, **kwargs):
    """Calculate the simple moving average for a column of a dataset

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

def sma_intercepts(dataframe, column, short_window, long_window):
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

    Returns:
        DataFrame with indices corresponding to dataframe
    """
    x_column = '_datetime_start'

    sma_short = calculate_sma(dataframe, x_column, column, window=short_window)
    sma_long = calculate_sma(dataframe, x_column, column, window=long_window)

    results = {
        x_column: [],
        'positive': [],
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
        above = above_now

    # Now convert sma index values to dataframe.index values
    x_series = dataframe[x_column]
    results['index'] = [x_series[x_series == x].index[0] for x in results[x_column]]

    return pandas.DataFrame(results).set_index('index')

def sma_local_minmax(dataframe, column, short_window, long_window, min_domain=3):
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
    }

    # Iterate through all the transition points and identify mins and maxes
    # between each intersection.  Note that we are ignoring the unbounded
    # first and last regions
    prev_row = None
    for row in intercepts.itertuples():
        if prev_row is not None:
            minmax_idx = None
            if prev_row.positive and not row.positive:
                minmax_idx = dataframe.loc[prev_row.Index:row.Index][column].idxmax()
            elif not prev_row.positive and row.positive:
                minmax_idx = dataframe.loc[prev_row.Index:row.Index][column].idxmin()
            if minmax_idx:
                results['index'].append(minmax_idx)
                results[x_column].append(dataframe.loc[minmax_idx][x_column])
                results['positive'].append(prev_row.positive)
                results[column].append(dataframe.loc[minmax_idx][column])
        prev_row = row

    return pandas.DataFrame(results).set_index('index')
