#!/bin/bash

. vars.sh


mkdir -pe "train" "val" "test"

for i in ${TRAIN_FILES[@]}; do
        screen -dmS "decompress_${i}" tar -xzf "./${i}" -C ./train
done

for i in ${VALIDATION_FILES[@]}; do
        screen -dmS "decompress_${i}" tar -xzf "./${i}" -C ./val
done

for i in ${TEST_FILES[@]}; do
        screen -dmS "decompress_${i}" unzip -P "${TEST_DECOMPRESSION_KEY}" "./${i}" -d ./test
done
