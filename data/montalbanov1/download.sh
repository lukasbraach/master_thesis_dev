#!/bin/bash

. vars.sh

for i in ${TRAIN_FILES[@]} ${VALIDATION_FILES[@]} ${TEST_FILES[@]}; do
        screen -dmS "download_$i" wget -c "http://158.109.8.102/2013/dataset/${i}"
done


