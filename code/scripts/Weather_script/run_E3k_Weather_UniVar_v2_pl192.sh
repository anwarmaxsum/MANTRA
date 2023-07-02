python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_E3k_96_192 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 3 \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.0001 \
  --dropout 0.01 \
  --anomaly 1.0 \
  --d_model 256 \
  --fix_seed 2023 \
  --checkpoints ./checkpoints1/