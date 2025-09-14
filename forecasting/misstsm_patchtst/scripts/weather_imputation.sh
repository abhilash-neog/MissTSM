ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
PRED_LEN_LIST=$5
use_multi_gpu=$6
use_misstsm=$7
IMPUTATION=$8

model_name=PatchTST

model_id_name=weather
data_name=weather

GT_ROOT_PATH="./forecasting_datasets/weather/"
root_path_name="./synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather_imputed_${IMPUTATION}.csv"

OUTPUT_PATH="./outputs/${IMPUTATION}/${MASKINGTYPE}/weather_v${TRIAL}/"
CHECKPOINT="./PatchTST_ckpts/${IMPUTATION}/checkpoints/"
seq_len=336

random_seed=2021

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u run_longExp.py \
          --random_seed $random_seed \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path weather.csv \
          --model_id $model_id_name_$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
          --e_layers 3 \
          --n_heads 16 \
          --d_model 128 \
          --q_dim 128 \
          --k_dim 8 \
          --v_dim 8 \
          --d_ff 256 \
          --dropout 0.2\
          --fc_dropout 0.2\
          --head_dropout 0\
          --patch_len 16\
          --stride 8\
          --des 'Exp' \
          --train_epochs 100\
          --gpu $DEVICES \
          --patience 20\
          --itr 1 \
          --batch_size 128 \
          --trial $TRIAL \
          --learning_rate 0.0001 \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH \
          --use_multi_gpu $use_multi_gpu \
          --misstsm $use_misstsm
    done
done