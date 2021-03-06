#!/bin/bash

source ../conf/variables.sh

FOLDER=${PATCH_PATH}     #/data/patches

PRED_VERSION=patch-level-6classes.txt

DIS_FOLDER=${HEATMAP_TXT_OUTPUT_FOLDER}
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
    echo -e "x_loc y_loc lepidic benign acinar micropap mucinous solid\n$(cat ${DIS_FOLDER}/${dis})" > ${DIS_FOLDER}/${dis}
done

PRED_VERSION=patch-level-color.txt
DIS_FOLDER=${HEATMAP_TXT_OUTPUT_FOLDER}
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "color-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
done

exit 0
