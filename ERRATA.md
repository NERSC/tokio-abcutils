A Year in the Life of a Parallel File System
================================================================================

October 16, 2018
--------------------------------------------------------------------------------

Figures 3 and 4 in "A Year in the Life of a Parallel File System" show
mislabeled dates for the Mira (mira-fs1) dataset.  For example, Fig. 4 shows no
data for Sept 15, 17, and 20 when in fact these blank dates should be Sept 12,
14, and 17.  This error does not affect any numerical results presented in the
paper.

In addition, the specific `_datetime_start` and `_datetime_end` timestamps
contained in the `summaries/mira-summaries_2017-02-14_2018-02-15.csv.gz`
dataset, as of October 16, 2018, do not match the epoch timestamps of the
`darshan_start_time` feature.  The value stored in `darshan_start_time` is the
ground-truth correct timestamp, and this can be verified using the Darshan logs
included in the full year-long dataset.

The incorrect values for `_datetime_start` and `_datetime_end` resulted from
converting the string representation of the jobs' start/end times (e.g.,
`2017-08-01 02:33:53`) into UTC-based timestamps assuming the strings were
localized to the `America/Los_Angeles` time zone instead of `America/Chicago`
time zone where those datetime strings were actually encoded.  This has no
effect on the numerical results presented in the paper and only affects the
specific way in which the Mira (mira-fs1) dataset was visualized in Figures 3
and 4.
