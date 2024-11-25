#!/usr/bin/env bash

: '
outsourced constants. load them with this command to get correct references
in pycharm:
source "$(dirname "$0")"/constants.sh || source ./constants.sh
if you dont care about pycharm references:
source "$(dirname "$0")"/constants.sh
'
# current repository where the shs subfolder is in
readonly REPO="~/poolscripts"

# scripts
readonly SSH_SAFE="./sshu.sh"
readonly POOLOUT_READER_PY="./pool.py"

# no other pools than valid pools will be checked (reduces server load)
# readonly VALID_POOLS=`seq 25 35; seq 37 46; seq 52 63`
readonly VALID_POOLS=`seq 0 10; seq 12 80`
readonly DEFAULT_MIN=1
readonly DEFAULT_MAX=70

# following are used to find free pools
readonly LOG_DIR="./_logs" # must match pool.py
readonly LOG_UPDATE="./_logs/poolupdate.log" # simple timestamp of last update
readonly LOG_OUT="./_logs/poolout.log" # pinfo.sh output per pool
readonly LOG_ERR="./_logs/poolerr.log" # pinfo.sh errors per pool
readonly LOG_STATUS="./_logs/poolstatus.log" # list of all free pools
readonly LOG_ACT="./_logs/poolact.log" # list of all running pools (free or not)
