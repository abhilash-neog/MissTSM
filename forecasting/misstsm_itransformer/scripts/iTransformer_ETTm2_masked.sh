export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
RUN_NAME=$5
MTSM_NORM=$6
EMBED=$7
LAYERNORM=$8
INVERTED=$9
SKIP=${10}
use_misstsm=${11}

GT_ROOT_PATH="./forecasting_datasets/ETT/"
root_path_name="./synthetic_datasets/ETTm2/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_ettm2.csv"

OUTPUT_PATH="./outputs_${RUN_NAME}/${MASKINGTYPE}/ETTm2_v${TRIAL}/"
CHECKPOINT="./iTransformer_ckpts/${RUN_NAME}/checkpoints/"
seq_len=336

model_name=iTransformer

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path ETTm2.csv \
          --model_id "ETTm2_${seq_len}_${pred_len}" \
          --model $model_name \
          --data ETTm2 \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --e_layers 2 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 128 \
          --q_dim 128 \
          --k_dim 8 \
          --v_dim 8 \
          --d_ff 128 \
          --itr 1 \
          --gpu $DEVICES \
          --trial $TRIAL \
          --embed_type $EMBED\
          --mtsm_norm $MTSM_NORM \
          --layernorm $LAYERNORM \
          --inverted $INVERTED\
          --skip_connection $SKIP \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH \
          --misstsm $use_misstsm
    done
done