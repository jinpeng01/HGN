export MODEL=xlnet-large-cased
epoch=20
lr=5e-5
wis=1qq3qq5qq7
data_type=W16_bio
connect_type=dot-att

    CUDA_VISIBLE_DEVICES=0 python run_hgn.py \
    --train_data_dir=data/$data_type/train_merge.txt \
    --dev_data_dir=data/$data_type/dev.txt \
    --test_data_dir=data/$data_type/test.txt \
    --bert_model=${MODEL} \
    --task_name=ner \
    --output_dir=./output/bert_multi_window \
    --max_seq_length=128 \
    --num_train_epochs ${epoch} \
    --do_train \
    --gpu_id 0 \
    --learning_rate ${lr} \
    --warmup_proportion=0.1 \
    --train_batch_size=32 \
    --use_bilstm \
    --use_multiple_window \
    --windows_list=${wis} \
    --connect_type=${connect_type}

