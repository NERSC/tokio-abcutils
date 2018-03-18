#!/usr/bin/env bash
#
# Find darshan log files generated by TOKIO-ABC and generate json summary files
# for each.  Parallelized with xargs because TOKIO's summarize_jobs.py is very
# slow.
# 

export jobhost=$1
if [ -z "$jobhost" ]; then
    echo -e "Syntax: $0 <cori|edison>\n" >&2
    echo "  NOTE: This script assumes that it is being run from a host that provides" >&2
    echo "  the sacct command, and that the job ids in Darshan logs are consistent" >&2
    echo "  with what this sacct command will provide.  In other words, do not run" >&2
    echo "  this script on Cori against Darshan logs generated on Edison!" >&2
    exit 1
fi

if [ "$jobhost" != "edison" -a "$jobhost" != "cori" ]; then
    echo "Unknown system [$jobhost]" >&2
    exit 1
fi

if [ -z "$NERSC_JOBSDB_HOST" ]; then
    echo "NERSC_JOBSDB_HOST is not set, so summarize_job.py will fail!" >&2
    exit 1
fi

export REPO_HOME="$(dirname $(readlink -f ${BASH_SOURCE[0]}))"
export output_dir="$REPO_HOME/summaries/$jobhost"

if [ ! -d "$output_dir" ]; then
    mkdir -vp "$output_dir"
fi

process() {
    for darshanlog in "$@"; do
        json_output="${output_dir}/$(basename $darshanlog .darshan).json"

        # if file doesn't exist, or if it does exist and is not nonzero
        if [ ! -f "$json_output" -o ! -s "$json_output" ]; then
            echo "Generating $json_output"
            $REPO_HOME/pytokio/bin/summarize_job.py --jobhost=$jobhost \
                                                    --concurrentjobs \
                                                    --topology=$REPO_HOME/data/${jobhost}.xtdb2proc.gz \
                                                    --ost \
                                                    --json \
                                                    $darshanlog > $json_output
#       else
#           echo "$json_output already exists with size $(stat --format=%s $json_output); skipping"
        fi
    done
}
export -f process

# Add -quit to the find command below to test changes (it will only process one
# darshan log before stopping)
find -L $REPO_HOME/results/runs.${jobhost}.* -name '*.darshan' -print0 | xargs -n 64 -P 16 -0 bash -c 'process "$@"' derp
