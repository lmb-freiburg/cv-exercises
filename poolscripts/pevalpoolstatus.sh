#!/usr/bin/env bash

: '
runs pool.py on poolout.log to fill poolstatus.log with active pools
'

# load constants with workaround for pycharm variable resolving
source "$(dirname "$0")"/constants.sh || source ./constants.sh

# flags
pyflag=""
while test $# != 0
do
    case "$1" in
        -h | -H | --help)
            echo "SHOWING HELP (-h) for $0"
            echo "runs shs/pool.py on file ${LOG_OUT} to produce statusfile ${LOG_STATUS}"
            echo "-p: verbose python script"
            echo "-v: output statusfile ${LOG_STATUS}"
            exit 1
        ;;
        -p) pyflag="-v" ;;
        -v) verbose="t" ;;
    esac
    shift
done

# evaluate pools file
user_me=$(whoami)
val=`python3 ${POOLOUT_READER_PY} --user ${user_me} --poolfile ${LOG_OUT} ${pyflag}`

# check error code
if [[ $? -ne 0 ]]; then
    # on login server use this python instead of venv
    val=`/usr/pkg/bin/python ${POOLOUT_READER_PY} --user ${user_me} --poolfile ${LOG_OUT} ${pyflag}`
fi

echo "$val" > ${LOG_STATUS}

if [[ "$verbose" == "t" ]]; then
    cat ${LOG_STATUS}
fi

# do it verbose
python3 ${POOLOUT_READER_PY} --user ${user_me} --poolfile ${LOG_OUT} -v