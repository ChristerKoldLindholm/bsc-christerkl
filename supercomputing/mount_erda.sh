#!/bin/bash
key="$HOME/.ssh/id_ed25519"
user="trh104@alumni.ku.dk"
erdadir="dryad_hydroacoustic_inglefield"
mnt="$HOME/erda_bsc_christerkl"
if [ -f "$key" ]
then mkdir -p ${mnt}
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${key}
else 
    echo "'${key}' is not an ssh key"
fi 
