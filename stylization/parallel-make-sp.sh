#!/bin/bash

set -e

# Start from directory of script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Make sure the datasets directory is symlinked
ln -sfn ../datasets data

# Only create a stylized validation set
for SPLIT in val # train
do

DATASET=data/places/${SPLIT}
OUTPUT=data/stylized-places/${SPLIT}
DIRS=$(ls -1 $DATASET)

# Loop through every class dir: this script can be run in parallel
for D in $DIRS
do

# Only one process should work on a directory at once
if [ ! -d "${OUTPUT}/${D}" ]
then

# If the process fails, then remove the whole directory
mkdir -p "${OUTPUT}/${D}"
trap "rm -rf ${OUTPUT}/${D}; exit" INT TERM EXIT

python3 stylize.py \
    --content-dir "${DATASET}/${D}" \
    --output-dir "${OUTPUT}/${D}" \
    --style-dir 'data/painter-by-numbers/train/' \
    --num-styles 1 \
    --content-size 0 \
    --style-size 256

# Mark the directory as complete
date > "${OUTPUT}/${D}/done.txt"
trap - INT TERM EXIT

fi

done

done
