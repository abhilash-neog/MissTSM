IMPUTATION=$1 #SAITS or spline
USE_MISSTSM=$2

DEVICES=0
DATASET=FCR
model_name=iTransformer

OUTPUT_PATH="./${model_name}/${IMPUTATION}/"
CHECKPOINT="./${model_name}/${IMPUTATION}/checkpoints/"

root_path_name="./realworld_data_MissTSM/${DATASET}"
data_path_name="${DATASET}_${IMPUTATION}.csv"

gt_root_path_name="./realworld_data_MissTSM/${DATASET}"
gt_data_path_name="${DATASET}_clean.csv"

seq_len=21
label_len=7

model_id_name=Lake
data_name=Lake
for pred_len in 28 35 42; do
    python -u run.py \
        --target 'daily_median_chla_interp_ugL' \
        --num_workers 0 \
        --freq d\
        --batch_size 32 \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --gt_root_path $gt_root_path_name \
        --gt_data_path $gt_data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --label_len $label_len \
        --e_layers 2 \
        --enc_in 13 \
        --dec_in 13 \
        --c_out 13 \
        --des 'Exp' \
        --d_model 128 \
        --q_dim 128 \
        --k_dim 8 \
        --v_dim 8 \
        --d_ff 128 \
        --train_epochs 10\
        --itr 1 \
        --gpu $DEVICES \
        --embed_type "tfi"\
        --mtsm_norm 0 \
        --layernorm 1 \
        --inverted 1\
        --skip_connection 1 \
        --trial 1 \
        --checkpoints $CHECKPOINT \
        --output_path $OUTPUT_PATH \
        --misstsm $USE_MISSTSM
done
