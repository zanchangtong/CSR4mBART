#!/bin/bash
DATA_BIN_DIR=/path/to/save
spm_data_dir=/path/to/spm_data

get_embedding(){
    TRAIN=train
    VALID=valid
    TEST=test

    language_pair=$1-$2
    SRC=$1
    TGT=$2
    DEST_dir=$DATA_BIN_DIR/${language_pair}

    cat ${spm_data_dir}/${TRAIN}.${language_pair}.spm.${SRC} ${spm_data_dir}/${VALID}.${language_pair}.spm.${SRC} > ${spm_data_dir}/corpus.${language_pair}.spm.${SRC}
    cat ${spm_data_dir}/${TRAIN}.${language_pair}.spm.${TGT} ${spm_data_dir}/${VALID}.${language_pair}.spm.${TGT} > ${spm_data_dir}/corpus.${language_pair}.spm.${TGT}
    
    ./word2vec/trunk/word2vec -train ${spm_data_dir}/corpus.${language_pair}.spm.${SRC} -output ${DEST_dir}/${language_pair}.s_embedding.txt \
    -cbow 0 -size 512 -window 10 -negative 10 -hs 0 -sample 1e- -threads 50 - binary 0 -min-count 5 -iter 10 -save-vocab ${DEST_dir}/w2v_vocab.${SRC} &

    ./word2vec/trunk/word2vec -train ${spm_data_dir}/corpus.${language_pair}.spm.${TGT} -output ${DEST_dir}/${language_pair}.t_embedding.txt \
    -cbow 0 -size 512 -window 10 -negative 10 -hs 0 -sample 1e- -threads 50 - binary 0 -min-count 5 -iter 10 -save-vocab ${DEST_dir}/w2v_vocab.${TGT} &
}

normalize(){
    language_pair=$1-$2
    SRC=$1
    TGT=$2
    DEST_dir=$DATA_BIN_DIR/${language_pair}
    python vecmap/normalize_embeddings.py unit center -i ${DEST_dir}/${language_pair}.s_embedding.txt -o ${DEST_dir}/s_embedding.normalized.txt &
    python vecmap/normalize_embeddings.py unit center -i ${DEST_dir}/${language_pair}.t_embedding.txt -o ${DEST_dir}/t_embedding.normalized.txt &
}

preprocess_ML50(){
        language_pair=$1-$2
        SRC=$1
        TGT=$2

        CUDA_VISIBLE_DEVICES=0 python vecmap/map_embeddings.py --orthogonal ${DEST_dir}/s_embedding.normalized.txt ${DEST_dir}/t_embedding.normalized.txt ${DEST_dir}/s_embedding.mapped.txt ${DEST_dir}/t_embedding.mapped.txt --init_numerals --self_learning -v --cuda
        python vecmap/cosine_d.py ${DEST_dir}/s_embedding.mapped.txt ${DEST_dir}/t_embedding.mapped.txt --output_dir ${DEST_dir}/
        python ML50_src_2_tgt.py ${DEST_dir}/cosine_d.npy ${DEST_dir}/ ${language_pair}
        pushd vecmap
        w2v_vocab_dir=${DEST_dir}
        translation_vocab_dir=${DEST_dir}
        python index_transform.py ${w2v_vocab_dir}/w2v_vocab.${SRC} ${w2v_vocab_dir}/w2v_vocab.${TGT} \
                        ${translation_vocab_dir}/dict.${SRC}.txt ${translation_vocab_dir}/dict.${TGT}.txt \
                                ${DEST_dir}/${language_pair}.npy ${DEST_dir}/${language_pair}
        popd

        language_pair=$2-$1
        SRC=$2
        TGT=$1
        CUDA_VISIBLE_DEVICES=1 python vecmap/map_embeddings.py --orthogonal ${DEST_dir}/s_embedding.normalized.txt ${DEST_dir}/t_embedding.normalized.txt ${DEST_dir}/s_embedding.mapped.txt ${DEST_dir}/t_embedding.mapped.txt --init_numerals --self_learning -v --cuda
        python vecmap/cosine_d.py ${DEST_dir}/s_embedding.mapped.txt ${DEST_dir}/t_embedding.mapped.txt --output_dir ${DEST_dir}/
        python ML50_src_2_tgt.py ${DEST_dir}/cosine_d.npy ${DEST_dir}/ ${language_pair}
        pushd vecmap
        w2v_vocab_dir=${DEST_dir}
        translation_vocab_dir=${DEST_dir}
        python index_transform.py ${w2v_vocab_dir}/w2v_vocab.${SRC} ${w2v_vocab_dir}/w2v_vocab.${TGT} \
                ${translation_vocab_dir}/dict.${SRC}.txt ${translation_vocab_dir}/dict.${TGT}.txt \
                ${DEST_dir}/${language_pair}.npy ${DEST_dir}/${language_pair}
        popd
}

SRC=$1
TGT=$2

get_embedding $SRC $TGT 
wait 
normalize $SRC $TGT 
wait 
preprocess_ML50 $SRC $TGT 
