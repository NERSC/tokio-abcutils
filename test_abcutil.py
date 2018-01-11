#!/usr/bin/env python
"""
Basic tests and demonstrations of functionality for abcutil module
"""

import pandas
import matplotlib
import abcutils

SAMPLE_INPUT = 'sample_summaries.csv'
SAMPLE_CORRELATE_WITH = 'darshan_agg_perf_by_slowest_posix'

# prevent the test from throwing DISPLAY errors
matplotlib.pyplot.switch_backend('agg')

class TestAbcDataFrame(object):
    """
    Test class for various abcutil dataframe functions
    """
    def __init__(self):
        """
        Always need to operate on a dataframe
        """
        self.dataframe = abcutils.load_and_synthesize_csv(SAMPLE_INPUT)

    def test_load_and_synthesize(self):
        """
        test abcutils.load_and_syntesize_csv functionality
        """
        for expected_col in ['_system',
                             'darshan_write_job?',
                             'darshan_app',
                             'coverage_factor_read_bw',
                             'coverage_factor_nodehrs',
                             'darshan_biggest_api_bytes',
                             'darshan_fpp_or_ssf_job',
                             'darshan_app_api']:
            print "Checking for column %s" % expected_col
            assert expected_col in self.dataframe.columns

        print "Asserting that performance > 0"
        assert (self.dataframe['darshan_agg_perf_by_slowest_posix'] > 0.0).all()
        print "Asserting that bandwidth coverage factor > 0"
        assert (self.dataframe['coverage_factor_bw'] > 0.0).all()
        print "Asserting that bandwidth coverage factor < 10"
        assert (self.dataframe['coverage_factor_bw'] < 10.0).all()
        print "Asserting that all records were either read or write jobs"
        assert ((self.dataframe['darshan_read_or_write_job'] == 'read') \
                | (self.dataframe['darshan_read_or_write_job'] == 'write')).all()

        # Also test the coverage_factor_nodehrs which depends on the configuration json
        print "Asserting that nodehrs coverage factor > 0"
        assert (self.dataframe['coverage_factor_nodehrs'] > 0.0).all()
        print "Asserting that nodehrs coverage factor < 10"
        assert (self.dataframe['coverage_factor_nodehrs'] < 1.0).all()

    def test_normalized_perf(self):
        """
        test abcutils.normalize_column
        """
        target_col = 'darshan_agg_perf_by_slowest_posix'
        group_by_cols = ['darshan_app',
                         '_file_system',
                         'darshan_fpp_or_ssf_job',
                         'darshan_read_or_write_job']
        new_col_base = 'darshan_normalized_perf'

        # modifies the dataframe in-place; returns nothing
        abcutils.normalize_column(
            dataframe=self.dataframe,
            target_col=target_col,
            group_by_cols=group_by_cols,
            new_col_base=new_col_base)

        assert (self.dataframe['darshan_normalized_perf_by_max'] > 0.0).all()
        assert (self.dataframe['darshan_normalized_perf_by_max'] <= 1.0).all()

    def test_filters(self):
        """
        test abcutils.apply_filters
        """
        filters = [
            # only want IOR jobs
            self.dataframe['darshan_app'] == 'ior',
            # only want jobs that did more than 1 TiB of I/O
            self.dataframe['darshan_biggest_api_bytes'] > 2**40,
        ]

        filtered_df = abcutils.apply_filters(self.dataframe, filters, verbose=True)
        assert len(filtered_df) < len(self.dataframe)
        assert len(filtered_df) > 0

    def test_plot_correlation_matrix(self):
        """
        test abcutils.plot.correlation_matrix
        """
        fig, correlations = abcutils.plot.correlation_matrix(self.dataframe)
        assert fig is not None
        assert correlations is not None
        num_correlations = len(fig.axes[0].get_xticklabels())
        assert num_correlations > 1
        assert num_correlations <= len(self.dataframe.columns) # non-numeric columns aren't included

    def test_correlation_calc_correlation_vector(self):
        """
        test abcutils.correlation.calc_correlation_vector
        """
        input_dataframe = pandas.DataFrame.from_dict({
            'time': [ 1, 2, 3, 4 ],
            'perf': [ 5.0, 3.0, 2.5, 2.75 ],
            'var1': [ 7, 12, 3, 6 ],
            'var2': [ 3, 8, 2, 9 ],
            'str1': [ 'hello', 'world', 'abc', 'def' ]
        })
        vector = abcutils.correlation.calc_correlation_vector(input_dataframe, 'perf')
        print vector

        print "len(vector) (%d) == len(input_dataframe.columns) - 2 (%d)" % (len(vector), len(input_dataframe.columns))
        assert len(vector) == len(input_dataframe.columns) - 2

    def test_plot_correlation_vector_table(self):
        """
        test abcutils.plot.correlation_vector_table
        """
        vector = abcutils.correlation.calc_correlation_vector(self.dataframe, SAMPLE_CORRELATE_WITH)
        fig = abcutils.plot.correlation_vector_table(vector)
        assert fig is not None
        cells_dict = fig.axes[0].tables[0].get_celld()


        row_labels = []
        for cell_pos, cell_obj in fig.axes[0].tables[0].get_celld().iteritems():
            if cell_pos[1] == -1:
                row_labels.append(cell_obj.get_text().get_text())

        for row_label in row_labels:
            print "%s in vector.index?" % row_label
            assert row_label in vector.index

        for row_label in vector.index:
            print "%s in row_labels?" % row_label
            assert row_label in row_labels

        # -1 because of the column heading cell
        print (len(cells_dict) - 1) / 2, len(vector), len(vector.index)
        assert (len(cells_dict) - 1) / 2 == len(vector)
