#!/bin/bash
KEY="$HOME/.ssh/erda_job"
USER="trh104@alumni.ku.dk"
REMOTE_DIR="/erda_bsc_christerkl/dryad_hydroacoustic_inglefield"
MNT="$HOME/erda_bsc_christerkl"
if [ -f "$KEY" ]
then 
    mkdir -p ${MNT}
    sshfs ${USER}@io.erda.dk:${REMOTE_DIR} ${MNT} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${KEY}
else 
    echo "'${KEY}' is not an ssh key"
fi
    # sshfs -o IdentitiesOnly=yes -o BatchMode=yes -o User="$USER" -o IdentityFile="$KEY" \
    #   -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    #   io.erda.dk:"$REMOTE_DIR" "$MNT"

# REMOTE_DIR="dryad_hydroacoustic_inglefield"
    # sshfs ${USER}@io.erda.dk:${REMOTE_DIR} ${MNT} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${KEY}