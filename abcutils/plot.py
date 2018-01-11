"""
Useful plotting routines for examining TOKIO-ABC data
"""
import matplotlib
import matplotlib.pyplot

def correlation_matrix(dataframe, labelsize=20, figsize=(20, 20), cmap='seismic'):
    """
    Function plots a graphical correlation matrix for each pair
    of columns in the dataframe.
    
    Input:
        dataframe: pandas DataFrame
        size: vertical and horizontal size of the plot
    """
    correlations = dataframe.corr()
    fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
    ax.matshow(correlations,
               cmap=matplotlib.pyplot.get_cmap(cmap),
               norm=matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0))

    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.columns)))
    ax.set_xticklabels(correlations.columns, rotation='vertical', fontsize=labelsize)
    ax.set_yticklabels(correlations.columns, fontsize=labelsize)
    return fig, correlations
