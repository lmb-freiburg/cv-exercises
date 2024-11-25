#!/usr/bin/env bash

: '
checks poolerr.log to find active pools into poolact.log
'

# load constants with workaround for pycharm variable resolving
source "$(dirname "$0")"/constants.sh || source ./constants.sh

# check flag passing
while test $# != 0
do
    case "$1" in
        -h | -H | --help)
            echo "SHOWING HELP (-h) for $0"
            echo "-v: verbose"
            exit 1
        ;;
        -v) verbose=t ;;
    esac
    shift
done

## find active pools from last error logs
cat ${LOG_ERR} | grep closed. | sed -e 's/Connec.*to //g' \
 | sed -e 's/ closed.//g' | tr '\n' ' ' > ${LOG_ACT}

# if verbose, output the active pools
if [[ "${verbose}" == "t" ]]; then
    cat ${LOG_ACT}
fi