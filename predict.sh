python run_race.py \
--data_dir=../ \
--bert_model=../bert-base-chinese \
--output_dir=../large_models \
--max_seq_length=320 \
--do_eval --do_lower_case \
--do_predict \
--train_batch_size=32 \
--eval_batch_size=4 \
--learning_rate=1e-5 \
--num_train_epochs=1 \
--gradient_accumulation_steps=8 \
--loss_scale=128