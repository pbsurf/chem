#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: `getpdb WXYZ`, where WXYZ is 4 letter PDB id"
fi

# PDB ftp is case-sensitive and uses lowercase; ${VAR,,} works in Bash 4+
PDB=${1,,}
wget -O "$1.pdb.gz" "ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/${PDB:1:2}/pdb$PDB.ent.gz" && gunzip "$1.pdb.gz"
