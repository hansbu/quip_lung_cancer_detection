#!/bin/bash

source ../conf/variables.sh

cd color
nohup bash color_stats.sh ${PATCH_PATH} 4 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_4.txt &
nohup bash color_stats.sh ${PATCH_PATH} 5 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_5.txt &

nohup bash color_stats.sh ${PATCH_PATH} 6 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_6.txt &
nohup bash color_stats.sh ${PATCH_PATH} 7 16 \
    &> ${LOG_OUTPUT_FOLDER}/log.color_stats_7.txt &

wait

exit 0
