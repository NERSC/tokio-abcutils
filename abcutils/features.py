import time
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
