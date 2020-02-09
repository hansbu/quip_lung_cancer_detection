#!/bin/bash

source ../conf/variables.sh

cd color
nohup bash color_stats.sh ${PATCH_PATH} 8 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_8.txt &
nohup bash color_stats.sh ${PATCH_PATH} 9 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_9.txt &

nohup bash color_stats.sh ${PATCH_PATH} 10 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_10.txt &
nohup bash color_stats.sh ${PATCH_PATH} 11 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_11.txt &

wait

exit 0
