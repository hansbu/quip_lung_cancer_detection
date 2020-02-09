#!/bin/bash

source ../conf/variables.sh

cd color
nohup bash color_stats.sh ${PATCH_PATH} 0 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_0.txt &
nohup bash color_stats.sh ${PATCH_PATH} 1 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_1.txt &

nohup bash color_stats.sh ${PATCH_PATH} 2 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_2.txt &

nohup bash color_stats.sh ${PATCH_PATH} 3 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_3.txt &

wait

exit 0
