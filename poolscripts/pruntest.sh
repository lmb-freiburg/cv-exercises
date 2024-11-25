#!/usr/bin/env bash

: '
run the pinfo.sh script on all poolpcs, echo results to stdout
'

# load constants with workaround for pycharm variable resolving
source "$(dirname "$0")"/constants.sh || source ./constants.sh

# check parameters
min=${DEFAULT_MIN}
max=${DEFAULT_MAX}
while test $# != 0
do
    case "$1" in
        -h | -H | --help)
            echo "SHOWING HELP (-h) for $0"
            echo "-l min max: test pools in this range only (default 01-70)"
            echo "-t: do not actually run the ssh commands"
            exit 1
        ;;
        -l) lim=t && min=$2 && max=$3 && shift && shift ;;
        -t) test=t ;;
    esac
    shift
done

# loop over pool pcs and run info script
shu='./sshu.sh'
echo starting ssh loop limits ${min} ${max}

for i in ${VALID_POOLS}; do
    # if pool out of given min max range, skip
    if (($i < $min || $i > $max)); then
        continue;
    fi

    # zeropad the pool number to get a valid pool string
    istr=`printf "%02d" ${i}`

    # output pool number and run info script
    echo -----POOLPC-----
    echo ${i}
    if [[ "$test" != "t" ]]; then
        ${shu} -t tfpool${i} ${REPO}/pinfo.sh
        sleep 1 # avoid server overload
    fi
done
