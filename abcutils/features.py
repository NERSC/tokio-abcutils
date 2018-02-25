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

def find_streaks(iterable=False, min_streak=3):
    """
    Find a streak of monotonically increasing or decreasing values

    Args:
        iterable: any enumerable object
        min_streak (int): The minimum number of consecutively monotonic values
            before a streak is reported
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
