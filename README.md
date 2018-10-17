# TOKIO Automated Benchmark Collection Utilities (tokio-abcutils)

## Setting Up

### Step 1. Prepare the working environment

We assume that all of the TOKIO-ABC results are stored in a subdirectory of this
repository called `results`:

    $ mkdir results
    $ cd results
    $ for i in /global/project/projectdirs/m888/glock/tokio-abc-results/runs.{cori,edison}.2017-*; do ln -vs $i;done

In order for the concurrent jobs metrics to be populated, you must define the
following environment variables:

    export NERSC_JOBSDB_HOST="..."
    export NERSC_JOBSDB_USER="..."
    export NERSC_JOBSDB_PASSWORD="..."
    export NERSC_JOBSDB_DB="..."

See a NERSC staff member for the correct values to gain access to the NERSC
jobs database.

### Step 2. Generate job summary files

Then we generate summary json files for each Darshan log.  This takes a
significant amount of time because it involves opening every Darshan log, then
collecting metrics from across the system that correspond to that job.  To do
this in parallel, use the included `parallel_summarize_job.sh` script, e.g.,

    $ ./parallel_summarize_job.sh edison 2>&1 | tee -a summarize_jobs-edison.log
    mkdir: created directory '/global/project/projectdirs/m888/glock/tokio-year/summaries/edison'
    Generating /global/project/projectdirs/m888/glock/tokio-year/summaries/edison/glock_ior_id3906633_2-14-63024-14939811182217632593_1.json
    Generating /global/project/projectdirs/m888/glock/tokio-year/summaries/edison/glock_ior_id4048967_2-19-64883-9509271909828150823_1.json
    Generating /global/project/projectdirs/m888/glock/tokio-year/summaries/edison/glock_ior_id4015752_2-18-63772-1376825852187540237_1.json
    ...

This script is just a parallel wrapper around `summarize_job.py` and is invoked
on each darshan log with options similar to the following:

    $ ./pytokio/bin/summarize_job.py --jobhost=cori \
                                     --concurrentjobs \
                                     --topology=data/cori.xtdb2proc.gz \
                                     --ost \
                                     --json \
                                     results/runs.cori.2017-12/runs.cori.2017-12-31.5/glock_dbscan_*.darshan

This `summarize_job.py` script retrieves and indexes data from each connector,
but it does not strive to synthesize cross-connector metrics such as coverage
factors.  That occurs in analysis that we will perform later on.

### Step 3. Collate job summaries

We then convert the collection of per-job summary json files into a normalized
collection of records in CSV format.

    $ ./normalize_job_summaries.py --output summaries/edison-summaries_%s.csv ./summaries/edison/*.json

The `normalize_job_summaries.py` script takes any number of json files generated
by `summarize_job.py`, finds all of the fields that were populated, and creates
a Pandas DataFrame from all of those records.  Each record that is missing one
or more keys from `summarize_job.py` simply has that field left as a NaN.

The `--output` argument allows you to specify a file name to which the
normalized data should be written in CSV format.  If the `--output` file name
contains a `%s`, this is replaced by the date range represented in the
normalized data.
