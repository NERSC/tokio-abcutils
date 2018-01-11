#!/usr/bin/env python
"""
Basic tests and demonstrations of functionality for abcutil module
"""

import abcutils

SAMPLE_INPUT = 'sample_summaries.csv'

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
