#!/usr/bin/env bash

: '
this script is run on a pool computer and outputs the required information.
uses a temporary directory in the user home folder
'
uname=`whoami`
c_dir="/home/$uname/poolscripts/_logs_pool"
mkdir ${c_dir} 2>/dev/null

sep_line=-----SEPARATOR-----

# host (tfpoolXX)
hostname
echo ${sep_line}
# users active
who
echo ${sep_line}

# gpu
nvidia-smi | grep MiB
