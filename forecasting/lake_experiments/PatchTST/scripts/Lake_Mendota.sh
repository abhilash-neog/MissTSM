IMPUTATION=$1 #SAITS or spline
USE_MISSTSM=$2

DEVICES=0
model_name=PatchTST
DATASET=Mendota


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
random_seed=2021

for pred_len in 28 35 42; do
    python -u ./run_longExp.py \
        --target 'avg_chlor_rfu' \
        --num_workers 0 \
        --freq d \
        --label_len 7 \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --gt_root_path $gt_root_path_name \
        --gt_data_path $gt_data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --label_len $label_len \
        --features M \
        --enc_in 12 \
        --dec_in 12 \
        --c_out 12 \
        --e_layers 3 \
        --n_heads 4 \
        --d_model 16 \
        --q_dim 16 \
        --k_dim 8 \
        --v_dim 8 \
        --d_ff 128 \
        --dropout 0.3 \
        --fc_dropout 0.3 \
        --head_dropout 0 \
        --patch_len 16 \
        --stride 8 \
        --des 'Exp' \
        --gpu $DEVICES \
        --mtsm_embed "tfi"\
        --mtsm_norm 0 \
        --layernorm 1 \
        --skip_connection 1 \
        --train_epochs 100 \
        --itr 1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --checkpoints $CHECKPOINT \
        --output_path $OUTPUT_PATH \
        --trial 1 \
        --misstsm $USE_MISSTSM
done