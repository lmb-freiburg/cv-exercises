#!/usr/bin/env bash

: '
save SSH command
'

# define ssh command
sshu='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5'

# run command
${sshu} "$@"