"""
Useful plotting routines for examining TOKIO-ABC data
"""
import time
import datetime
import pandas
import numpy
import matplotlib
import matplotlib.pyplot
import tokio.tools.umami
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

def correlation_vector_table(dataframe, ax=None, fontsize=14, col_name_map=None, row_name_map=None, **kwargs):
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

def generate_umami(dataframe, plot_metrics, highlight_index=-1):
    """
    Generate a full UMAMI diagram based on an input dataframe and a list of
    column names.  Relies on abcutil.CONFIG to create labels and determine if
    big_is_good
    """
    umami = tokio.tools.umami.Umami()
    for metric in plot_metrics:
        label = abcutils.CONFIG['metric_labels'].get(metric, metric)
        big_is_good = abcutils.CONFIG['metric_big_is_good'].get(metric, True)
        umami[metric] = tokio.tools.umami.UmamiMetric(
            timestamps=dataframe['_datetime_start'],
            values=dataframe[metric],
            label=label,
            big_is_good=big_is_good)

    return umami.plot(highlight_index=highlight_index)

def clustered_correlation_bars(dataframes, plot_metrics, column_key=None, width=0.5, ax=None):
    """
    Generate clustered bar graph of correlation coefficients
    
    Generates a clustered bar graph where each cluster represents a
    correlation, and the bars within each cluste represent the correlation as
    calculated over a subset of the total dataset.  The number of clusters is
    len(plot_metrics) and the number of bars in each cluster is
    len(dataframes.keys()).
    
    Args:
        dataframes (dict of DataFrame): dictionary of dataframes generated by
            abcutils.correlation.calc_correlation_vector
        plot_metrics (list of str): keys corresponding to index values of
            dataframes; each specified key will result in a cluster of
            bar charts
        column_key (str): the prefix (if any) of the column name to plot.  If
            None, look for the first column name that ends in "coefficient" and
            "p-value"
        width (float): width of each bar to be passed to matplotlib.pyplot.bar
        ax (Axes): axes on which graph should be generated        
    """
    if ax is None:
        fig, ax = matplotlib.pyplot.subplots()
        fig.set_size_inches(16, 4)
    ax.grid()
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Correlation with Performance")

    xticks = []
    xticklabels = []

    # identify the columns of interest.  note we make the implicit assumption
    # that all dataframes in `dataframes` have the same columns here!
    target_coeff_col = None
    target_pval_col = None
    first_df = dataframes.itervalues().next()
    if column_key is None:
        for col_name in first_df.columns:
            if col_name.endswith(" coefficient"):
                column_key = col_name.rsplit(None, 1)[0]
                print col_name, column_key
                if ("%s p-value" % column_key) in first_df.columns:
                    break
                else:
                    print "[%s] not in %s" % ("%s p-value" % column_key, first_df.columns)
    target_coeff_col = "%s coefficient" % column_key
    target_pval_col = "%s p-value" % column_key
    if target_coeff_col not in first_df.columns \
    or target_pval_col not in first_df.columns:
        raise KeyError("Columns %s/%s not in dataframes" % (target_coeff_col, target_pval_col))

    # For each metric, generate a cluster of bars
    for index, plot_metric in enumerate(plot_metrics):
        xticks.append(float(index)*width*len(dataframes.keys()))
        xticklabels.append(plot_metric)
        color = 'black' if index % 2 == 0 else 'gold'
        
        # plot each box in a cluster of three as an individual plt.box() call so
        # that we can color the box edges independently and according to their
        # p-value
        for bindex, bin_key in enumerate(sorted(dataframes.keys())):
            pval = dataframes[bin_key][target_pval_col].loc[plot_metric]
            if pval < 0.01:
                edgecolor = 'blue'
            elif pval < 0.05:
                edgecolor = 'green'
            else:
                edgecolor = 'red'
            ax.bar(bindex * width + xticks[-1],
                   dataframes[bin_key][target_coeff_col].loc[plot_metric],
                   width=width*0.95,
                   color=color,
                   edgecolor=edgecolor,
                   align='edge')

    ax.set_xticks(xticks)
    ax.set_xticklabels([abcutils.CONFIG['metric_labels'].get(x, x) for x in xticklabels], rotation=30, ha='right')

    return ax
