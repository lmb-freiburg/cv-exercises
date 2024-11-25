#!/usr/bin/env bash

: '
checks which pools are active, parses some input and calls pruntest.sh
to test the pool. then prints some output

-f: force update
-o: force no update (output only)
-v: verbose
-l: range test (default 01-70)
'

# load constants with workaround for pycharm variable resolving
source "$(dirname "$0")"/constants.sh || source ./constants.sh

# check flag passing
min=1
max=70
while test $# != 0
do
    case "$1" in
        -h | -H | --help)
            echo "SHOWING HELP (-h) for $0"
            echo "checks which pools are active. default rechecks after 1 hour."
            echo "-l min max: test pools in this range only (default 01-70)"
            echo "-f: force pool update"
            echo "-o: force NO pool update"
            echo "-v: verbose"
            exit 1
        ;;
        -l) lim=t && min=$2 && max=$3 && shift && shift ;;
        -f) force=t ;;
        -o) output_only=t ;;
        -v) verbose=t ;;
    esac
    shift
done

# determine files
mkdir ${LOG_DIR} 2>/dev/null

# check time difference since last update
secs=`cat ${LOG_UPDATE} 2> /dev/null || echo 0`
now=`date +'%s'`
diff=`expr ${now} - ${secs}`
mdiff=3600
minsdiff=`expr ${diff} / 60`

# verbose output
if [[ "$verbose" == "t" ]]; then
    echo "limits $min $max minutes difference $minsdiff"
fi

# update if time over, forced by flag or files dont exist
# gets disabled by output_only
if { [[ ${diff} -ge ${mdiff} ]] || [[ "$force" == "t" ]] \
 || [[ ! -f "$LOG_OUT" ]] || [[ ! -f "$LOG_ERR" ]]; } \
 && [[ "$output_only" != "t" ]]; then
    echo ${now} > ${LOG_UPDATE}

    if [[ "$verbose" == "t" ]]; then
        cat ${LOG_STATUS}
        echo "updating... (can take up to several minutes)"
        verbosestr="-v"
    fi

    # write pool pc status to file (takes 1-2 min)
    ./pruntest.sh ${verbosestr} -l ${min} ${max} > ${LOG_OUT} 2> ${LOG_ERR}
else
    if [[ "$verbose" == "t" ]]; then
        echo "not updating, last update was $minsdiff min ago"
    fi
fi

# evaluate free pools with python script
./pevalpoolstatus.sh

# evaluate active pools from error log
./pevalpoolactive.sh

# cat "$LOG_DIR/poolstatus.log"
