
generate_best(){
  checkpoint_dir=$1
  SRC=$2
  TGT=$3
  cuda_id=$4
  output_dir=$5
  TEST=test
  language_pair=${SRC}-$TGT  # xx2en
  DEST=$6
  echo "SRC: $SRC" 
  echo ">> $language_pair evluate: best ..."

  echo "cuda:$cuda_id"
  CUDA_VISIBLE_DEVICES=$cuda_id fairseq-generate $DEST \
    --path $checkpoint_dir/checkpoint_best.pt \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    -t ${TGT} -s ${SRC} \
    --batch-size 60 --langs $langs > $output_dir/${language_pair}_best
}

generate_last(){
  checkpoint_dir=$1
  SRC=$2
  TGT=$3
  cuda_id=$4
  output_dir=$5
  TEST=test
  language_pair=${SRC}-$TGT  # xx2en
  DEST=$6
  echo ">> $language_pair evluate: last ..."

  echo "cuda:$cuda_id"
  CUDA_VISIBLE_DEVICES=$cuda_id fairseq-generate $DEST \
    --path $checkpoint_dir/checkpoint_last.pt \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    -t ${TGT} -s ${SRC} \
    --batch-size 60 --langs ${langs} > $output_dir/${language_pair}_last
}

generate_avg5(){
  checkpoint_dir=$1
  SRC=$2
  TGT=$3
  cuda_id=$4
  output_dir=$5
  TEST=test
  language_pair=${SRC}-$TGT  # xx2en
  DEST=$6
  echo ">> $language_pair evluate: avg5 ..."
  
  if [ ! -f $checkpoint_dir/checkpoint_avg5.pt ];then
    python average_checkpoints.py \
        --inputs $checkpoint_dir \
        --num-update-checkpoints 5 \
        --output $checkpoint_dir/checkpoint_avg5.pt
  fi

  CUDA_VISIBLE_DEVICES=$cuda_id fairseq-generate $DEST \
    --path $checkpoint_dir/checkpoint_avg5.pt \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    -t ${TGT} -s ${SRC} \
    --batch-size 60 --langs $langs > $output_dir/${language_pair}_avg5
}


langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

MODEL=/path/to/mbart.cc25/sentence.bpe.model
spm_decode=/path/to/sentencepiece/build/src/spm_decode
export LC_ALL=C.UTF-8 

checkpoint_dir=$1
SRC=$2
TGT=$3
cuda_id=$4
language_pair=$SRC-$TGT
DATA_BIN=/path/to/data-in/ML50/$language_pair
output=$checkpoint_dir/bilingual_output.re
mkdir $output

generate_best ${checkpoint_dir} $SRC $TGT $cuda_id $output $DATA_BIN &
generate_last ${checkpoint_dir} $SRC $TGT $cuda_id $output $DATA_BIN &
generate_avg5 ${checkpoint_dir} $SRC $TGT $cuda_id $output $DATA_BIN &

wait 
BLEU_result=$output/bleu.results

# change language id to target language id
printf "${language_pair}.best_BLEU, " >> $BLEU_result
cat ${output}/${language_pair}_best | grep -P "^H" |sort -V |cut -f 3- |${spm_decode} --model=${MODEL} | sed 's/\[de_DE\]//g' > ${output}/${language_pair}_best.hyp
cat ${output}/${language_pair}_best | grep -P "^T" |sort -V |cut -f 2- |${spm_decode} --model=${MODEL} | sed 's/\[de_DE\]//g' > ${output}/${language_pair}_best.ref
sacrebleu ${output}/${language_pair}_best.ref -i ${output}/${language_pair}_best.hyp --language-pair "${SRC: 0: 2}-${TGT: 0: 2}" -b -w 1 >> $BLEU_result
wait
printf "${language_pair}.last_BLEU, " >> $BLEU_result
cat ${output}/${language_pair}_last | grep -P "^H" |sort -V |cut -f 3- |${spm_decode} --model=${MODEL} | sed 's/\[de_DE\]//g' > ${output}/${language_pair}_last.hyp
cat ${output}/${language_pair}_last | grep -P "^T" |sort -V |cut -f 2- |${spm_decode} --model=${MODEL} | sed 's/\[de_DE\]//g' > ${output}/${language_pair}_last.ref
sacrebleu ${output}/${language_pair}_last.ref -i ${output}/${language_pair}_last.hyp --language-pair "${SRC: 0: 2}-${TGT: 0: 2}" -b -w 1 >> $BLEU_result
wait
printf "${language_pair}.avg5_BLEU, " >> $BLEU_result
cat ${output}/${language_pair}_avg5 | grep -P "^H" |sort -V |cut -f 3- |${spm_decode} --model=${MODEL} | sed 's/\[de_DE\]//g' > ${output}/${language_pair}_avg5.hyp
cat ${output}/${language_pair}_avg5 | grep -P "^T" |sort -V |cut -f 2- |${spm_decode} --model=${MODEL} | sed 's/\[de_DE\]//g' > ${output}/${language_pair}_avg5.ref
sacrebleu ${output}/${language_pair}_avg5.ref -i ${output}/${language_pair}_avg5.hyp --language-pair "${SRC: 0: 2}-${TGT: 0: 2}" -b -w 1 >> $BLEU_result


