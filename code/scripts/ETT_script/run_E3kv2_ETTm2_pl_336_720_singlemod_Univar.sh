python -u run_singlemodU.py  \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id E3k_ETTm2_Singlemod_UniVarr_96_336 \
  --model B6autoformer \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --n_learner 3 \
  --urt_head 2 \
  --learning_rate 0.001 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 \
  --fix_seed 2023 \
  --dropout 0.01 \
  --anomaly 1.0 \
  --d_model 256 \
  --checkpoints ./checkpoints2/


python -u run_singlemodU.py  \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id E3k_ETTm2_Singlemod_UniVarr_96_720 \
  --model B6autoformer \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --n_learner 3 \
  --urt_head 2 \
  --learning_rate 0.001 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 1 \
  --fix_seed 2023 \
  --dropout 0.1 \
  --d_model 256 \
  --anomaly 2.0 \
  --checkpoints ./checkpoints3/

#train_epochs 1 \

