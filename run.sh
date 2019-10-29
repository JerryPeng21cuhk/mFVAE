#!/bin/bash
# 2018-2019 jerrypeng

# This script implements mFVAE.
# For the details, pls refer to paper: 
#   mixture factorized auto-encoder for unsupervised hierarchical deep factorization of speech signal

# Speaker verification experiment is carried on VoxCeleb 1.
# There is no data augmentation

# mFAE is implemented in file: local/mfvae/model.py
# You can make slight revisions of:
##  local/mfvae/train_mfvae.py
##  local/mfvae/extract_embeddings.py
##  local/mfvae/extract_posteriors.py
#  in order to carry out experments of mFAE.

## Besides, some pykaldi executables are implemented to calculate the
##   stats of posteriors. They are in local/pykaldi-bin directory.

## The codes in this repository are the clean-up of voxceleb-tdnn-s2pretrain.
## voxceleb-tdnn-s2pretrain implements:
##    data augmentation, 
##    appending i-vector as input to frame tokenizer
##    FHVAE discriminative loss for utterance embedder
##    pairwise classification loss on embeddings
##    conditional feature reconstruction for ZeroSpeech 2017
##    ApplyCmvnSliding(This seems problematic, need further investigation)
##    read_gmm_model for GMM-CNN embedder.
##    multi-head attention for mFAE
##    VAD mask for regression loss
##    self-supervsied speaker embeddings
##    mFVAE without reparameterization trick on categorical distribution 
##                                    (slow and results in GPU mem overflow)
##
## I delete all these unnecessary functions in this repository.


set -e
set +o posix



root_data_dir=/lan/ibdata/SPEECH_DATABASE/voxceleb1
stage=1
log_filename="temp.log"
modelname="mfvae"

gpu_idxs="\"\""
mfccdir=`pwd`/mfcc
cluster_num=128
embed_dim=600
bnf_feat_dim=64
feat_dim=30
resume_epoch=0
num_epochs=50
gmvn_apply=True
gmvn_norm_vars=True
# cmvn_reverse=False
gmvn_stats_rxfilename=data/train/gmvn_stats
path2model=exp/$modelname/model/${num_epochs}.mdl
trials=data/voxceleb1_trials_sv
mkdir -p exp/$modelname/log

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ -f exp/$modelname/log/report_${log_filename} ] && echo "! Exists exp/$modelname/log/report_${log_filename}" && tail -vn4 exp/$modelname/log/report_${log_filename} && exit 0;

if [ $stage -le 0 ]; then  
  local/make_voxceleb1_sv.pl $root_data_dir data
  echo ">> make features ---"
  for i in data/train data/test; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$cpu_cmd" $i exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $i
  done

  compute-cmvn-stats scp:data/train/feats.scp data/train/cmvn_stats
fi

if [ $stage -le 1 ]; then
  # local/fix_feats_dur.sh --nj 40 --feats-dur 5 --raw-feats true data/train exp/feats_5s/train || exit 1;
  python local/mfvae/train_mfvae.py --batch-size 64 \
                              --num-epochs $num_epochs \
                              --gpu-idxs ${gpu_idxs} \
                              --resume-epoch ${resume_epoch} \
                              --log-step 500 \
                              --init-lr 1e-3 \
                              --end-lr 1e-4 \
                              --data-crop True \
                              --sort-utts True \
                              --frame-num-thresh 4000 \
                              --cluster-num ${cluster_num} \
                              --embed-dim ${embed_dim} \
                              --feat-dim ${feat_dim} \
                              --bnf-feat-dim ${bnf_feat_dim} \
                              --gmvn-apply ${gmvn_apply} \
                              --gmvn-norm-vars ${gmvn_norm_vars} \
                              --gmvn-stats-rxfilename ${gmvn_stats_rxfilename} \
                              --train-data-dir data/train \
                              --log-dir exp/$modelname/log \
                              --log-filename "train_${log_filename}" \
                              --model-save-dir exp/$modelname/model
fi

if [ $stage -le 2 ]; then
  for i in train test; do
    python local/mfvae/extract_embeddings.py --batch-size 1 \
                                       --data-crop False \
                                       --gpu-idxs ${gpu_idxs} \
                                       --frame-num-thresh 4000 \
                                       --cluster-num ${cluster_num} \
                                       --embed-dim ${embed_dim} \
                                       --bnf-feat-dim ${bnf_feat_dim} \
                                       --feat-dim ${feat_dim} \
                                       --gmvn-apply ${gmvn_apply} \
                                       --gmvn-norm-vars ${gmvn_norm_vars} \
                                       --gmvn-stats-rxfilename "${gmvn_stats_rxfilename}" \
                                       --ipath2model "${path2model}" \
                                       --log-dir exp/$modelname/log \
                                       --log-filename "extract_${i}_embeddings_${log_filename}" \
                                       --embed-dir exp/$modelname/${i}_embed_vectors
  done
fi

if [ $stage -le 3 ]; then
  # Average the utterance-level xvectors to get speaker-level embeddings.
  echo "$0: computing mean of embeddings for each speaker"
  $cmd exp/$modelname/train_embed_vectors/log/speaker_mean.log \
    ivector-mean ark:data/train/spk2utt scp:exp/$modelname/train_embed_vectors/embeddings.scp \
    ark,scp:exp/$modelname/train_embed_vectors/spk_embeddings.ark,exp/$modelname/train_embed_vectors/spk_embeddings.scp ark,t:exp/$modelname/train_embed_vectors/num_utts.ark || exit 1;

  # Compute the gloabl mean vector for centering the evaluation embeddings.
  $cpu_cmd exp/$modelname/train_embed_vectors/log/compute_mean.log \
    ivector-mean scp:exp/$modelname/train_embed_vectors/embeddings.scp \
    exp/$modelname/train_embed_vectors/mean.vec || exit 1;

  lda_dim=150
  $cpu_cmd exp/$modelname/train_embed_vectors/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/$modelname/train_embed_vectors/embeddings.scp ark:- |" \
    ark:data/train/utt2spk exp/$modelname/train_embed_vectors/transform.mat || exit 1;

  $cpu_cmd exp/$modelname/train_embed_vectors/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean exp/$modelname/train_embed_vectors/mean.vec scp:exp/$modelname/train_embed_vectors/embeddings.scp ark:- | transform-vec exp/$modelname/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/$modelname/train_embed_vectors/plda || exit 1;
fi

if [ $stage -le 4 ]; then
  # do cosine scoring
  $cpu_cmd exp/$modelname/scores/log/cosine_scoring.log \
    cat $trials \| awk '{print $1" "$2}' \| \
    ivector-compute-dot-products - \
      "ark:ivector-subtract-global-mean exp/$modelname/train_embed_vectors/mean.vec scp:exp/$modelname/test_embed_vectors/embeddings.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean exp/$modelname/train_embed_vectors/mean.vec scp:exp/$modelname/test_embed_vectors/embeddings.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      exp/$modelname/scores/cosine_scores || exit 1;

  eer=$(paste $trials exp/$modelname/scores/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Cosine scoring, EER: ${eer}%"
  echo "Time: $(date)." > exp/$modelname/log/eerScores_${log_filename}
  echo "Cosine scoring, EER: ${eer}%" > exp/$modelname/log/eerScores_${log_filename}
fi

if [ $stage -le 5 ]; then

  # do lda + cosine scoring
  $cpu_cmd exp/$modelname/scores/log/lda_scoring.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-subtract-global-mean exp/$modelname/train_embed_vectors/mean.vec scp:exp/$modelname/test_embed_vectors/embeddings.scp ark:- | ivector-transform exp/$modelname/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/$modelname/train_embed_vectors/mean.vec scp:exp/$modelname/test_embed_vectors/embeddings.scp ark:- | ivector-transform exp/$modelname/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/$modelname/scores/lda_scores || exit 1;
    
  eer=$(paste $trials exp/$modelname/scores/lda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "LDA + cosine scoring, EER: ${eer}%"
  echo "LDA + cosine scoring, EER: ${eer}%" >> exp/$modelname/log/eerScores_${log_filename}

fi

if [ $stage -le 6 ]; then
  # Get result using the out-of-domain PLDA model.
  $cpu_cmd exp/$modelname/scores/log/plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
      "ivector-copy-plda --smoothing=0.0 exp/$modelname/train_embed_vectors/plda - |" \
      "ark:ivector-subtract-global-mean exp/$modelname/train_embed_vectors/mean.vec scp:exp/$modelname/test_embed_vectors/embeddings.scp ark:- | transform-vec exp/$modelname/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean exp/$modelname/train_embed_vectors/mean.vec scp:exp/$modelname/test_embed_vectors/embeddings.scp ark:- | transform-vec exp/$modelname/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$trials' | cut -d\  --fields=1,2 |" exp/$modelname/scores/plda_scores || exit 1;

  eer=$(paste $trials exp/$modelname/scores/plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: ${eer}%"
  echo "Using Out-of-Domain PLDA, EER: ${eer}%" >> exp/$modelname/log/eerScores_${log_filename}

fi

for i in train extract_train_embeddings extract_test_embeddings eerScores; do
  cat exp/$modelname/log/${i}_${log_filename}
done >> exp/$modelname/log/report_${log_filename}

