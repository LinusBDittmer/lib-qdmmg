#!/bin/bash
#
#SBATCH -o "%x.o%j"
#SBATCH -J base
#SBATCH --time="01:00:00"
#SBATCH --mem=11010048K
#SBATCH --mail-type=NONE
#SBATCH -N 1
#SBATCH -c 4

#
###################################
#
SUBMIT_HOST=$SLURM_SUBMIT_HOST
SUBMIT_SERVER=$SLURM_SUBMIT_HOST
SUBMIT_QUEUE=$WHO_CARES
SUBMIT_WORKDIR=$SLURM_SUBMIT_DIR
JOBID=$SLURM_JOB_ID
JOBNAME=$SLURM_JOB_NAME
QUEUE=$WHO_CARES
O_PATH=$NOT_IMPLEMENTED
O_HOME=$NOT_IMPLEMENTED
NODES=$SLURM_JOB_NODELIST
NODES_UNIQUE=$(echo "$NODES" | sort -u)
RETURN_VALUE=0
NODE_WORKDIR="/scratch/ccprak16/base_$SLURM_JOB_ID"
NODE_SCRATCHDIR="/lscratch/ccprak16/base_$SLURM_JOB_ID"
#
###################################
#
 print_info() {
    echo ------------------------------------------------------
    echo "Job is running on nodes"
    echo "$NODES" | sed 's/^/    /g'
    echo ------------------------------------------------------
    echo qsys: job was submitted from $SUBMIT_HOST
    echo qsys: originating queue is $SUBMIT_QUEUE
    echo qsys: executing queue is $QUEUE
    echo qsys: original working directory is $SUBMIT_WORKDIR
    echo qsys: job identifier is $JOBID
    echo qsys: job name is $JOBNAME
    echo qsys: current home directory is $O_HOME
    echo qsys: PATH = $O_PATH
    echo ------------------------------------------------------
    echo
}

stage_in() {
    rm -f "$SUBMIT_WORKDIR/job_not_successful"

    echo "Calculation working directory: $NODE_WORKDIR"
    echo "            scratch directory: $NODE_SCRATCHDIR"

    # create workdir and cd to it.
    if ! mkdir -m700 -p $NODE_SCRATCHDIR $NODE_WORKDIR; then
        echo "Could not create scratch($NODE_SCRATCHDIR) or workdir($NODE_WORKDIR)" >&2
        exit 1
    fi
    cd $NODE_WORKDIR

    echo
    echo ------------------------------------------------------
    echo
}

stage_out() {
    if [ "$RETURN_VALUE" != "0" ]; then
        touch "$SUBMIT_WORKDIR/job_not_successful"
    fi

    echo
    echo ------------------------------------------------------
    echo

    echo "Final files in $SUBMIT_WORKDIR:"
    (
        cd $SUBMIT_WORKDIR
        ls -l | sed 's/^/    /g'
    )

    echo
    echo "More files can be found in $NODE_WORKDIR and $NODE_SCRATCHDIR on"
    echo "$NODES_UNIQUE" | sed 's/^/    /g'
    echo
    echo "Sizes of these files:"

    if echo "$NODE_SCRATCHDIR"/* | grep -q "$NODE_SCRATCHDIR/\*$"; then
        # no files in scratchdir:
        du -shc * | sed 's/^/    /g'
    else
        du -shc * "$NODE_SCRATCHDIR"/* | sed 's/^/    /g'
    fi

    echo
    echo "If you want to delete these, run:"
    for node in $NODES_UNIQUE; do
        echo "    ssh $node rm -r \"$NODE_WORKDIR\" \"$NODE_SCRATCHDIR\""
    done
}

handle_error() {
    # Make sure this function is only called once
    # and not once for each parallel process
    trap ':' 2 9 15

    echo
    echo "#######################################"
    echo "#-- Early termination signal caught --#"
    echo "#######################################"
    echo
    error_hooks
    stage_out
}

payload_hooks() {
:
for FILEORDIRPATH in $SLURM_SUBMIT_DIR/base.in; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$SLURM_SUBMIT_DIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$NODE_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$NODE_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $SLURM_SUBMIT_DIR/potential.pot; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$SLURM_SUBMIT_DIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$NODE_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$NODE_WORKDIR/$DIR"
    fi
done


export QCSCRATCH="$NODE_SCRATCHDIR"
/export/home/ccprak16/dreuwBin/qchem/versions/qchem-6.0 -nt 4 "base.in" "base.out"
RETURN_VALUE=$?

# check if job terminated successfully
if ! tail -n 30 "base.out" | grep -q "Thank you very much for using Q-Chem.  Have a nice day."; then
    RETURN_VALUE=1
fi

for FILEORDIRPATH in $NODE_WORKDIR/base.out; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/base.in.fchk; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/plots; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/base.out.plots; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/cap_adc_*_*.data; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/Epsilon.data; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/PEQS.data; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done


}

error_hooks() {
:
for FILEORDIRPATH in $NODE_WORKDIR/base.out; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$SLURM_SUBMIT_DIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$SLURM_SUBMIT_DIR/$DIR"
    fi
done

}
#
###################################
#
# Run the stuff:

print_info
stage_in

# If catch signals 2 9 15, run this function:
trap 'handle_error' 2 9 15

payload_hooks
stage_out
exit $RETURN_VALUE
