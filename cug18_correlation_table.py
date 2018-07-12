
# coding: utf-8

# In[ ]:



# In[ ]:



# In[ ]:

import os
import time
import datetime
import pandas
import numpy
import scipy.stats
import abcutils
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


# ## Global Analysis Constants

# In[ ]:

TEST_PLATFORMS = [
    'scratch1@edison',
#   'scratch2@edison',
    'scratch3@edison',
    'cscratch@cori-knl',
    'mira-fs1@mira'
]


# ## Load and Synthesize Data from CSV

# In[ ]:

filtered_df = abcutils.sc18paper.load_dataset()


# ## Global Correlation Table

# Show the most compelling correlations across all data.  This will be messy because it includes all file systems and test conditions, so there are many uncontrolled variables represented.

# In[ ]:

pandas.options.display.max_rows = 40

correlation = abcutils.correlation.calc_correlation_vector(filtered_df, correlate_with='darshan_normalized_perf_by_max')

filtered_correlations = abcutils.apply_filters(correlation, [correlation['p-value'] < 1.0e-5], verbose=True)
filtered_correlations.sort_values('coefficient')


# In[ ]:

ax = abcutils.plot.correlation_vector_table(filtered_correlations, row_name_map=abcutils.CONFIG['metric_labels'])
ax.get_figure().set_size_inches(4, 0.4 * len(filtered_correlations))


# Now draw the entire correlation table split out by _test platform_--a combination of the file system being tested and the node configuration being used to test it.

# In[ ]:

correlations = None
grouped_df = filtered_df.groupby('_test_platform')
for fs in TEST_PLATFORMS:
    # generate a single file system's correlation vector
    correlation = abcutils.correlation.calc_correlation_vector(
        grouped_df.get_group(fs),
        correlate_with='darshan_normalized_perf_by_max')
            
    # rename the columns in this vector to include the file system name
    new_cols = {}
    for index, col_name in enumerate(correlation.columns):
        new_cols[col_name] = "%s %s" % (fs, col_name)
    correlation.rename(columns=new_cols, inplace=True)
    
    # join the vector to the previous vectors' dataframe
    if correlations is None:
        correlations = correlation
    else:
        correlations = pandas.concat([correlations, correlation], axis='columns')


# In[ ]:

# Construct filter to show any metric that registered a low p-value for _any_ file system
filters = None
for fs in TEST_PLATFORMS:
    subfilter = correlations['%s p-value' % fs] < 1.0e-5
    if filters is None:
        filters = subfilter
    else:
        filters |= subfilter

ax = abcutils.plot.correlation_vector_table(
    correlations[filters],
    row_name_map=abcutils.CONFIG['metric_labels'])

# Set the table width larger if displaying lots of metrics
ax.get_figure().set_size_inches(20, 0.4 * len(correlations[filters]))


# In[ ]:

def cell_styler(cell_obj, coeff, pval):
    if pval < 1.0e-5:
        cell_obj.get_text().set_fontweight('bold')
    else:
        cell_obj.get_text().set_color('#00000099')
    set_color = (matplotlib.cm.get_cmap('YlGnBu'))(abs(coeff) / 1.0)

    cell_obj.set_color(set_color)


# In[ ]:

col_order = [
    'ior_fpp_write',
    'hacc_io_write_fpp_write',
    'ior_fpp_read',
    'hacc_io_read_fpp_read',
#   'ior_shared_write',
#   'vpicio_uni_shared_write',
#   'ior_shared_read',
#   'dbscan_read_shared_read'
]

good_counters = [
    'coverage_factor_bw',
    'coverage_factor_opens',
#   'coverage_factor_stats',
    'fs_ave_mds_cpu',
    'fs_ave_oss_cpu',
    'fs_max_mds_cpu',
    'fs_max_oss_cpu',
    'fshealth_ost_most_full_pct',
    'fshealth_ost_overloaded_oss_count',
    'topology_job_avg_radius',
]

correlations = None

apply_filters = filtered_df['_test_platform'] == 'cscratch@cori-knl'
#apply_filters &= filtered_df['coverage_factor_stats'] > 0.0
#apply_filters &= numpy.isfinite(filtered_df['coverage_factor_stats'])
#apply_filters &= filtered_df['coverage_factor_bw'] != 0.0
#apply_filters &= filtered_df['coverage_factor_opens'] != 0.0
#apply_filters &= filtered_df['_datetime_start'] >= datetime.datetime(2017, 8, 1)
#apply_filters &= filtered_df['_datetime_start'] < datetime.datetime(2018, 1, 1)

input_df = filtered_df[apply_filters][good_counters + ['_benchmark_id', 'darshan_normalized_perf_by_max']]

grouped_df = input_df.groupby('_benchmark_id')

for fs in col_order:
    # generate a single file system's correlation vector
    correlation = abcutils.correlation.calc_correlation_vector(
        grouped_df.get_group(fs),
        correlate_with='darshan_normalized_perf_by_max')
            
    # rename the columns in this vector to include the file system name
    new_cols = {}
    for index, col_name in enumerate(correlation.columns):
        new_cols[col_name] = "%s %s" % (fs, col_name)
    correlation.rename(columns=new_cols, inplace=True)
    
    # join the vector to the previous vectors' dataframe
    if correlations is None:
        correlations = correlation
    else:
        correlations = pandas.concat([correlations, correlation], axis='columns')
        
# Construct filter to show any metric that registered a low p-value for _any_ file system

filters = [True] * len(correlations)
#for fs in input_df['_benchmark_id'].unique():
#   subfilter = correlations['%s p-value' % fs] < 1.0e-5
#   if filters is None:
#       filters = subfilter
#   else:
#       filters |= subfilter

ax = abcutils.plot.correlation_vector_table(
    correlations.loc[good_counters],
    fontsize=18,
    row_name_map={
        'coverage_factor_bw': "Coverage Factor (Bandwidth)",
        "coverage_factor_opens": "Coverage Factor (opens)",
        "fs_ave_mds_cpu": "Average MDS CPU Load",
        "fs_ave_oss_cpu": "Average OSS CPU Load",
        "fs_max_mds_cpu": "Peak MDS CPU Load",
        "fs_max_oss_cpu": "Peak OSS CPU Load",
        "fshealth_ost_most_full_pct": "OST Fullness",
        "fshealth_ost_overloaded_oss_count": "Number of failed-over OSTs",
        "topology_job_avg_radius": "Average Job Radius",
    },
    col_name_map={
        'ior_fpp_write coefficient': 'IOR Write',
        'hacc_io_write_fpp_write coefficient': "HACC Write",
        'ior_fpp_read coefficient': "IOR Read",
        "hacc_io_read_fpp_read coefficient": "HACC Read",
    },
    cell_styler=cell_styler)

# Set the table width larger if displaying lots of metrics
ax.get_figure().set_size_inches(10, 5)
ax.get_figure().savefig('correlation_table_fpp_writes.pdf', bbox_inches='tight')


# In[ ]:



