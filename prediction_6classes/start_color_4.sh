#!/bin/bash

source ../conf/variables.sh

cd color
nohup bash color_stats.sh ${PATCH_PATH} 12 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_12.txt &
nohup bash color_stats.sh ${PATCH_PATH} 13 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_13.txt &

nohup bash color_stats.sh ${PATCH_PATH} 14 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_14.txt &
nohup bash color_stats.sh ${PATCH_PATH} 15 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_15.txt &

wait

exit 0
