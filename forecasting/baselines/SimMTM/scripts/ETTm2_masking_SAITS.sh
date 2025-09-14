BASE_PATH="./synthetic_datasets/ETTm2/"
ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
PRED_LEN_LIST=$5

DATA_PATH="v${TRIAL}_${MASKINGTYPE}_ettm2_imputed_SAITS.csv"

GT_ROOT_PATH="./forecasting_datasets/ETT/"

PRETRAIN_CHECKPOINTS_DIR="./SimMTM_ckpts/SAITS/"
FINETUNE_CHECKPOINTS_DIR="./SimMTM_ckpts/SAITS/"

OUTPUT_PATH="./outputs/SAITS/${MASKINGTYPE}/ETTm2_v${TRIAL}/"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do

    root_path="${BASE_PATH}${id}"
    python -u run.py \
        --task_name pretrain \
        --root_path $root_path \
        --data_path $DATA_PATH \
        --model_id ETTm2 \
        --model SimMTM \
        --data ETTm2 \
        --features M \
        --seq_len 336 \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 16 \
        --positive_nums 2 \
        --mask_rate 0.5 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 50 \
        --trial $TRIAL \
        --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
        --gpu $DEVICES

    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u run.py \
            --task_name finetune \
            --is_training 1 \
            --gt_root_path $GT_ROOT_PATH \
            --root_path $root_path \
            --data_path $DATA_PATH \
            --gt_data_path ETTm2.csv \
            --model_id ETTm2 \
            --model SimMTM \
            --data ETTm2 \
            --features M \
            --seq_len 336 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --n_heads 8 \
            --d_model 8 \
            --d_ff 16 \
            --dropout 0 \
            --batch_size 64 \
            --gpu $DEVICES \
            --trial $TRIAL \
            --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
            --checkpoints $FINETUNE_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH \
            --train_epochs 10
    done
done