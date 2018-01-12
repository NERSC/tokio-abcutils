"""
Useful plotting routines for examining TOKIO-ABC data
"""
import pandas
import matplotlib
import matplotlib.pyplot

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

#   if figsize is None:
#       figsize = (4, 0.4 * len(dataframe))

    ax = _init_ax(ax)

    # identify columns that contain correlation coefficients
    coefficient_keys = [x for x in dataframe.columns if 'coefficient' in x]

    ### the index is column -1
    table = pandas.plotting.table(ax,
                                dataframe[coefficient_keys],#.reindex(print_order),
                                loc='upper right',
                                colWidths=[0.8, 0.8, 3.8],
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
            remap_values[cell_pos] = col_name_map.get(value, value)
            height_scale = 2.0
        elif j == -1: # index cell
            remap_values[cell_pos] = row_name_map.get(value, value)
            cell_obj._loc = 'right'
        else:         # coefficient cell
            index = cells_dict[(i,-1)].get_text().get_text()
            column = cells_dict[(0,j)].get_text().get_text()
            cell_obj._loc = 'center'
            
            if value == "nan":
                cell_obj.set_color('grey')
                remap_values[cell_pos] = ""
                cell_obj.set_alpha(0.25)
            else:
                coeff = float(value)
                pval = dataframe.loc[index][column.replace('correlation', 'p-value')]

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
        cells_dict[cell_pos].get_text().set_text(new_value)

    return ax
