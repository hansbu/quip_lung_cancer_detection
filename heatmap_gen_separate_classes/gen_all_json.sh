#!/bin/bash

source ../conf/variables.sh
#HEATMAP_VERSION="lung-john-3c_lepidic_20200115_hanle"
#HEATMAP_TXT_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_txt_3classes_separate_class/heatmap_txt_lepidic

HEATMAP_VERSION=${1}
HEATMAP_TXT_OUTPUT_FOLDER=${2}

for files in ${HEATMAP_TXT_OUTPUT_FOLDER}/prediction-*; do
    if [[ "$files" == *.low_res* ]]; then
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-low_res  ${SVS_INPUT_PATH} lym 0.5 necrosis 0.5 &>> ${LOG_OUTPUT_FOLDER}/log.gen_json_multipleheat.low_res.txt
    else
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-high_res ${SVS_INPUT_PATH} lym 0.5 necrosis 0.5 &>> ${LOG_OUTPUT_FOLDER}/log.gen_json_multipleheat.txt    # HEATMAP_VERSION: neu_v1...
    fi
done

exit 0

