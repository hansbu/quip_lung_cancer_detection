#!/bin/bash

source ../conf/variables.sh

cd tumor_pred
nohup bash pred_thread_lym.sh \
    ${PATCH_PATH} 2 3 ${LYM_CNN_PRED_DEVICE} \
    &> ${LOG_OUTPUT_FOLDER}/log.pred_thread_tumor_2.txt &
    
wait
exit 0
