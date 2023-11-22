#!/bin/bash

dataset_dir=/path/to/dataset
PRETRAIN=/path/to/mbart.cc25/model.pt
exp=/path/to/save

source_lang=$2
target_lang=$3
language_pair=$2-$3
SRC=$2
TGT=$3
stage_1=${4:-'5000'}
stage_2=${5:-'30000'}
competence_step=${stage_1}
span_length=-1
update_freq=${6:-'1'}
gpu_id=${7:-'0,1,2,3'}

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

echo ">> stage_1 steps: ${stage_1}"
mkdir -p $exp/stage_1
CUDA_VISIBLE_DEVICES=$gpu_id fairseq-train $dataset_dir \
    --save-dir $exp/stage_1 \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task translation_from_pretrained_bart \
    --source-lang $source_lang --target-lang $target_lang \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --competence-step $competence_step \
    --total-num-update ${stage_1} --max-update ${stage_1} \
    --span-length ${span_length} \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 2048 --update-freq ${update_freq} \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 5 --no-epoch-checkpoints \
    --seed 222 --log-format simple --log-interval 2 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --restore-file $PRETRAIN \
    --langs $langs \
    --fp16 --num-workers 0 \
    --ddp-backend no_c10d \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe 'sentencepiece' \
    --eval-bleu-print-samples \
    --tensorboard-logdir $exp/tensorboard_1stage >> $exp/${language_pair}.log

echo "stage 2..."
competence_step=-1
PRETRAIN=$exp/stage_1/checkpoint_last.pt
CUDA_VISIBLE_DEVICES=$gpu_id fairseq-train $dataset_dir \
    --save-dir $exp \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task translation_from_pretrained_bart \
    --source-lang $source_lang --target-lang $target_lang \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --competence-step $competence_step \
    --total-num-update ${stage_2} --max-update ${stage_2} \
    --span-length ${span_length} \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 2048 --update-freq ${update_freq} \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 5 --no-epoch-checkpoints \
    --seed 222 --log-format simple --log-interval 2 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --restore-file $PRETRAIN \
    --langs $langs \
    --fp16 --num-workers 0 \
    --ddp-backend no_c10d \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe 'sentencepiece' \
    --eval-bleu-print-samples \
    --tensorboard-logdir $exp/tensorboard_2stage >> $exp/${language_pair}.log

