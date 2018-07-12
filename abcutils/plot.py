"""
Useful plotting routines for examining TOKIO-ABC data
"""
import re
import time
import datetime
import pandas
import numpy
import matplotlib
import matplotlib.pyplot
import tokio.analysis.umami
import abcutils

DEFAULT_BOXPLOT_GROUP_BY = ['darshan_fpp_or_ssf_job', 'darshan_read_or_write_job', 'darshan_app']
DEFAULT_BOXPLOT_SETTINGS = {
    'boxprops': {'linewidth': 2},
    'medianprops': {'linewidth': 4},
    'whiskerprops': {'linewidth': 2},
    'capprops': {'linewidth': 2},
    'widths': 0.75,
    'whis': [5, 95],
    'showfliers': False,
}

def _init_ax(ax):
    """
    Ensure that a fig, ax exists if not specified
    """
    if ax is None:
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111)

    return ax

def correlation_matrix(dataframe, ax=None, fontsize=20, cmap='seismic', **kwargs):
    """
    Plot graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        dataframe: pandas DataFrame
        fontsize: size of text labels
        cmap: string name of a matplotlib color map
    """
    correlations = dataframe.corr()
    ax = _init_ax(ax)
    ax.matshow(correlations,
               cmap=matplotlib.pyplot.get_cmap(cmap),
               norm=matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0),
               **kwargs)

    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.columns)))
    ax.set_xticklabels(correlations.columns, rotation='vertical', fontsize=fontsize)
    ax.set_yticklabels(correlations.columns, fontsize=fontsize)
    return ax, correlations


def _style_correlation_cell(cell_obj, coeff, pval):
    """Apply styling to a correlation table cell based on its value

    Args:
        cell_obj (matplotlib.table.Cell): the cell to be styled
        coeff (float): the value of the correlation coefficient
        pval (float): the value of the correlation's p-value
    """
    ### make moderate correlations **bold**
    if abs(coeff) >= 0.30:
        cell_obj.get_text().set_fontweight('bold')
    ### make weak correlations _italic_
    elif abs(coeff) < 0.10:
        cell_obj.get_text().set_fontstyle('italic')

    ### color code cells based on p-value
    if pval < 0.01:
        set_color = 'blue'
    elif pval < 0.05:
        set_color = 'green'
    else:
        set_color = 'red'

    ### for debugging, since the resulting figure doesn't contain any p-values
    # print "%30s pval=%10.4f; setting color to %s" % (index, pval, set_color)
    cell_obj.set_color(set_color)
    cell_obj.set_alpha(0.25)

def correlation_vector_table(dataframe,
                             ax=None,
                             fontsize=14,
                             col_name_map=None,
                             row_name_map=None,
                             cell_styler=_style_correlation_cell,
                             **kwargs):
    """
    Generate a table from a dataframe which contains columns containing
    'coefficient' and 'p-value' in their headers.  Intended to be used with the
    output of correlation.calc_correlations; will NOT work with the output of
    plot.correlation_matrix.

    col_name_map is a dict to convert dataframe's column names into textual labels for the table
    row_name_map is a dict to convert dataframe's index names into textual labels for the table
    """
    if col_name_map is None:
        col_name_map = {}
    if row_name_map is None:
        row_name_map = {}

    ax = _init_ax(ax)

    # identify columns that contain correlation coefficients
    coefficient_keys = [x for x in dataframe.columns if 'coefficient' in x]

    ### the index is column -1
    table = pandas.plotting.table(ax,
                                  dataframe[coefficient_keys],#.reindex(print_order),
                                  loc='upper right',
#                                 colWidths=[0.8, 0.8, 3.8],
                                  bbox=[0, 0, 1, 1],
                                  **kwargs)
    table.set_fontsize(fontsize)
    ax.axis('tight')
    ax.axis('off')

    ### Rewrite the contents of the table that Pandas gave us
    cells_dict = table.get_celld()
    remap_values = {}
    for cell_pos, cell_obj in cells_dict.iteritems():
        i, j = cell_pos
        value = cell_obj.get_text().get_text()
        height_scale = 1.0
        if i == 0:    # column headers
            remap_values[cell_pos] = col_name_map.get(value, value).replace(' ', '\n')
            height_scale = 2.0
        elif j == -1: # index cell
            remap_values[cell_pos] = row_name_map.get(value, value)
            cell_obj._loc = 'right'
        else:         # coefficient cell
            index = cells_dict[(i, -1)].get_text().get_text()
            column = cells_dict[(0, j)].get_text().get_text()
            cell_obj._loc = 'center'

            if value == "nan":
                cell_obj.set_color('grey')
                remap_values[cell_pos] = ""
                cell_obj.set_alpha(0.25)
            else:
                coeff = float(value)
                pval = dataframe.loc[index][column.replace('coefficient', 'p-value')]
                cell_styler(cell_obj, coeff, pval)
                remap_values[cell_pos] = "%+.4f" % coeff
        cell_obj.set_height(height_scale * cell_obj.get_height())
        cell_obj.set_edgecolor('black')

    ### Actually rewrite the cells now
    for cell_pos, new_value in remap_values.iteritems():
        text = cells_dict[cell_pos].get_text()
        text.set_text(new_value)
        # this helps when drawing gigantic tables for some reason
        text.set_size(fontsize)

    return ax

def default_rename_boxplot_label(current_label):
    """
    Convert a default boxplot axis label into a human-comprehensible one
    """
    # current_label = "(fpp, write, hacc_io_write)"
    try:
        fpp_or_shared, read_or_write, app = current_label.split(',')
    except ValueError:
        return current_label
    fpp_or_shared = fpp_or_shared[1:]
    read_or_write = read_or_write.strip()
    app = app[:-1].strip()

    if "ior" in app:
        if "shared" in fpp_or_shared:
            new_label = "IOR/shared"
        else:
            new_label = "IOR/fpp"
    else:
        new_label = abcutils.CONFIG['app_name_map'].get(app, app)

    if 'write' in read_or_write:
        new_label += "(W)"
    else:
        new_label += "(R)"

    return new_label

def grouped_boxplot(dataframe, plot_metric, group_by=DEFAULT_BOXPLOT_GROUP_BY,
                    rename_label_func=default_rename_boxplot_label, ax=None,
                    **kwargs):
    """
    Create a boxplot to show the distribution of one column that is grouped by a
    list of other columns.
    """
    other_settings = DEFAULT_BOXPLOT_SETTINGS.copy()
    other_settings.update(kwargs)

    if ax is None:
        _, ax = matplotlib.pyplot.subplots()

    dataframe.boxplot(column=[plot_metric],
                      by=group_by,
                      ax=ax,
                      **(other_settings))

    ax.set_title("")
    ax.get_figure().suptitle("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.grid(True)

    # relabel the x axis labels
    new_labels = []
    for axis_label in ax.get_xticklabels():
        axis_label.set_rotation(90)
        new_labels.append(rename_label_func(axis_label.get_text()))
    ax.set_xticklabels(new_labels)

    # subsequent analyses are free to override these ticks
    ax.yaxis.set_ticks(numpy.linspace(0.0, 1.0, 6))
    ax.set_ylim([-0.1, 1.1])

    return ax


def distribution_over_time(dataframe, plot_metric, date_start, date_end, date_delta=datetime.timedelta(days=7)):
    """
    Divide a dataframe of time series data into bins so that distributions
    within bins can be calculated over time, then return a list of bins,
    a list of binned series, and a list of human-readable bin labels.

    Args:
        dataframe (pandas.DataFrame): dataframe to divide into bins
        plot_metric (str): column name to divide into bins
        date_start (datetime.datetime): leading edge of first bin (inclusive)
        date_end (datetime.datetime): trailing edge of last bin (exclusive)
        date_delta (datetime.timedelta): width of each bin

    Returns:
        x (list of float): epoch-seconds corresponding to leading edge of bin
        y (list of pandas.Series): subset of dataframe falling into the bin
        x_labels (list of str): human-readable labels for each element of x
    """
    def increment_month(date):
        now_month = date.month
        now_year = date.year
        next_month = now_month + 1 if now_month < 12 else 1
        next_year = now_year + 1 if now_month == 12 else now_year
        return date.replace(year=next_year, month=next_month)

    x = []
    x_labels = []
    y = []
    date = date_start

    while date < date_end:
        # next_date = increment_month(date)
        next_date = date + date_delta
        y.append(dataframe[(dataframe['_datetime_start'] >= date) & (dataframe['_datetime_start'] < next_date)][plot_metric])
        x.append(time.mktime(date.timetuple()))
        x_labels.append(date.strftime("%b %d, %Y"))

        date = next_date
    return x, y, x_labels

def timeseries_boxplot(dataframe, plot_metric, date_start, date_end,
                       date_delta=datetime.timedelta(days=7), ax=None, **kwargs):
    """
    Create a boxplot to show the distribution of one column that is grouped by a
    list of other columns.

    Args:
        dataframe (pandas.DataFrame): dataframe to divide into bins
        plot_metric (str): column name to divide into bins
        date_start (datetime.datetime): leading edge of first bin (inclusive)
        date_end (datetime.datetime): trailing edge of last bin (exclusive)
        date_delta (datetime.timedelta): width of each bin
        ax (matplotlib.axes.Axes): optional axes to draw in (default: create new
            axes)
        kwargs (dict): additional styling parameters to pass to matplotlib's
            boxplot function

    Returns:
        matplotlib.axes.Axes containing the generated boxplot
    """
    other_settings = DEFAULT_BOXPLOT_SETTINGS.copy()
    other_settings['widths'] = date_delta.total_seconds() * 5 / 7
    other_settings.update(kwargs)

    if ax is None:
        _, ax = matplotlib.pyplot.subplots()

    x, y, x_labels = distribution_over_time(dataframe, plot_metric, date_start, date_end, date_delta)
    ax.boxplot(y, positions=[xx + date_delta.total_seconds() / 2 for xx in x], **other_settings)

    ax.set_ylim(0)
    ax.yaxis.grid(True)
    ax.set_title("")
    ax.set_xticklabels(x_labels, rotation=90)
    ax.get_figure().suptitle("")
    ax.get_figure().subplots_adjust(hspace=0.05, wspace=0.05)

    ax.xaxis.grid(True)

    return ax

def timeseries_manylines(lines, colorfunc=None, ax=None, **kwargs):
    """
    Plot a set of lines on a time series plot

    Args:
        lines (list of tuples of lists): data to draw.  each tuple contains
            (x, y) where x, y are lists of x and y values belonging to one line.
        colorfunc (function): takes a single element of `lines` and returns a
            color that can be passed to matplotlib.plot()
        ax (matplotlib.axes.Axes): optional axes to draw in (default: create new
            axes)
        kwargs (dict): additional styling parameters to pass to matplotlib's
            plot function
    """
    other_settings = {
        'marker': '',
        'linestyle': '-',
        'linewidth': 4,
        'markersize': 5,
    }
    other_settings.update(kwargs)

    if ax is None:
        _, ax = matplotlib.pyplot.subplots()

    for x, y in lines:
        if colorfunc:
            color = colorfunc((x, y))
        else:
            color = 'black'
        ax.plot(x, y, color=color, markerfacecolor=color, **other_settings)

    return ax

def timeseries_streaks(dataframe, streaks, ax=None, **kwargs):
    """
    Plot a set of streaks on a time series plot

    Args:
        dataframe (pandas.DataFrame): dataframe to which the streaks parameter refers
        streaks (list): output of abcutils.features.find_streaks_df()
        ax (matplotlib.axes.Axes): optional axes to draw in (default: create new axes)
        kwargs (dict): additional styling parameters to pass to matplotlib's plot function
    """
    colorfunc = lambda x: 'green' if x[1][-1] > x[1][0] else 'red'
    lines = []
    for streak in streaks:
        x = [time.mktime((dataframe.loc[x]['_datetime_start']).timetuple()) for x in streak[0]]
        lines.append((x, streak[1]))

    return timeseries_manylines(lines, colorfunc, ax, **kwargs)

def generate_umami(dataframe, plot_metrics, highlight_index=-1, show_empty=False, show_invariant=False):
    """
    Generate a full UMAMI diagram based on an input dataframe and a list of
    column names.  Relies on abcutil.CONFIG to create labels and determine if
    big_is_good
    """
    umami = tokio.tools.umami.Umami()
    for metric in plot_metrics:
        try:
            num_nans = sum(numpy.isnan(dataframe[metric]))
        except TypeError:
            num_nans = 0

        if (len(dataframe[metric].unique()) == 1) and (not show_invariant):
            continue

        if (len(dataframe[metric]) == num_nans) and (not show_empty):
            continue

        label = abcutils.CONFIG['metric_labels'].get(metric, metric)
        big_is_good = abcutils.CONFIG['metric_big_is_good'].get(metric, True)
        umami[metric] = tokio.tools.umami.UmamiMetric(
            timestamps=dataframe['_datetime_start'],
            values=dataframe[metric],
            label=label,
            big_is_good=big_is_good)

    return umami.plot(highlight_index=highlight_index)

def sma_overlaps(dataframe, plot_metric, short_window=None, long_window=None,
                 sma_intercepts=None, ax=None, 
                 regioncolors=['#0000000A', '#FFFFFF00'],
                 plotraw=True,
                 **kwargs):
    """Plot the raw data and region boundaries of a time series
    
    Args:
        dataframe (pandas.DataFrame): the filtered data from which loci are drawn
        plot_metric (str): column name corresponding to metric of interest in `dataframe`
        short_window: (int): size of the smaller simple moving average window
        long_window: (int): size of the larger simple moving average window
        sma_intercepts (pandas.DataFrame): output of 
            abcutils.features.sma_intercepts; if none is provided, calculate it
            using `short_window` and `long_window`
        ax (matplotlib.Axes): axes to draw on; create new Axes if None
        regioncolors (list of str): colors to use when shading alternating
            regions
        plotraw (bool): plot raw performance data in addition to SMAs
        kwargs: arguments to be passed to abcutils.features.sma_intercepts()
        
    Returns:
        matplotlib.Axes corresponding to drawn plot
    """
    x_raw = dataframe['_datetime_start'].apply(lambda x: long(time.mktime(x.timetuple()) / 86400) * 86400 )
#   x_raw = dataframe['_datetime_start'].apply(lambda x: time.mktime(x.timetuple()))
    y_raw = dataframe[plot_metric]

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = matplotlib.pyplot.subplots()
        fig.set_size_inches(16, 4)
        
    # plot the raw data
    delta = 86400 # hardcode sampling rate at 1 day
    if plotraw:
        ax.bar(x_raw, y_raw, width=delta, alpha=0.5)
    ax.grid()
    ax.set_ylabel(abcutils.CONFIG['metric_labels'].get(plot_metric, plot_metric).replace(" ", "\n"))
    ax.set_xticklabels([datetime.datetime.fromtimestamp(x).strftime("%b %d") for x in ax.get_xticks()])

    ### plot the intercept points demarcating different regions
    if sma_intercepts is None:
        sma_intercepts = abcutils.features.sma_intercepts(dataframe, plot_metric, short_window, long_window, **kwargs)
    left_x = None
    min_y, max_y = ax.get_ylim()
    for index, row in enumerate(sma_intercepts.itertuples()):
        this_x = abcutils.core.pd2epoch(row[1])
        if left_x is None:
            left_x = this_x
        else:
            ax.add_patch(matplotlib.patches.Rectangle(xy=(left_x, min_y),
                         width=(this_x - left_x),
                         height=(max_y - min_y),
                         color=regioncolors[index % 2],
                         linewidth=0))
            left_x = this_x

    ### also calculate and plot the SMAs
    if short_window:
        sma_short = abcutils.features.calculate_sma(dataframe, '_datetime_start', plot_metric, short_window, **kwargs)
        x_sma_short = [abcutils.core.pd2epoch(x) for x in sma_short.index]
        y_sma_short = sma_short.values
        ax.plot(x_sma_short, y_sma_short, color='C1', linewidth=2)

    if long_window:
        sma_long = abcutils.features.calculate_sma(dataframe, '_datetime_start', plot_metric, long_window, **kwargs)
        x_sma_long = [abcutils.core.pd2epoch(x) for x in sma_long.index]
        y_sma_long = sma_long.values
        ax.plot(x_sma_long, y_sma_long, color='C2', linewidth=2)

    return ax

def fix_xticks_timeseries(ax, format="%b %d, %Y", criteria=(lambda x: x.day == 1), **kwargs):
    """Set x tick markers of a time series plot
    
    Args:
        ax (matplotlib.Axes): plot axes to modify
        format (str): format for tick labels to be passed to strftime
        criteria (function): function to determine when a tick label should be
            drawn; its only argument should be an epoch timestamp (e.g., a value
            on the x axis)
        kwargs: arguments to be passed to `Axes.set_xticklabels()`
    """
    default_args = {
        'rotation': 30,
        'ha': 'right',
    }
    default_args.update(kwargs)
    xticks = []
    x_now, x_high = ax.get_xlim()
    while x_now < x_high:
        if criteria(datetime.datetime.fromtimestamp(x_now)): # include first of each month
            xticks.append(x_now)
        x_now += datetime.timedelta(days=1).total_seconds()
    xticklabels = [re.sub(r'(\D)0([\d]+)', r'\1\2', datetime.datetime.fromtimestamp(x).strftime(format), 1) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **default_args)
    ax.set_yticks(ax.get_yticks()[0:])

def draw_region(ax, region, **kwargs):
    """Create a shaded patch for a DataFrame of regions
    
    Args:
        ax (matplotlib.Axes): plot axes to modify
        region (pandas.DataFrame): one element of the output of
            `features.intercepts_to_regions()`
        kwargs: arguments to pass to `matplotlib.Axes.add_patch()`
    """
    args = {
        'facecolor': 'black',
        'linewidth': 0,
        'alpha': 0.10,
    }
    args.update(kwargs)
    min_y, max_y = ax.get_ylim()
    min_x = abcutils.core.pd2epoch(region.iloc[0]['_datetime_start'])
    max_x = abcutils.core.pd2epoch(region.iloc[-1]['_datetime_start'])
    patch = ax.add_patch(matplotlib.patches.Rectangle(xy=(min_x, min_y),
                         width=(max_x - min_x),
                         height=(max_y - min_y),
                         **args))
    return patch
