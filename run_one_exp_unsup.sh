#!/bin/bash
# 2018 jerrypeng

# This script try the modified resnet34 on voxceleb1 
# No data augmentation, no pretraining
# It serves as the implementation of paper:
#   Exploring the Encoding Layer and loss function in End-to-end
# 	speaker and language recognition system

set -e
set +o posix

featdir=`pwd`/fbank
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
testdir=test
trials=data/voxceleb1_trials_sv
stage=1

gpu_idxs="1"
cluster_num=128
embed_dim=600
bnf_feat_dim=64
latent_dim=60
feat_dim=30
resume_epoch=0
log_filename="temp.log"
cmvn_apply_global=True
cmvn_norm_vars=True
cmvn_reverse=False
cmvn_stats_rxfilename=data/train/cmvn_stats

path2model=exp/unsup_vae/50.mdl

mode=train #debug
nnscore=false

# log_filename="NoDimReduct_bnfDim${bnf_feat_dim}_clustNum${cluster_num}_clustDownsamp${clust_downsample}_featDownsamp${feat_downsample}.log"
mkdir -p exp/unsup_vae/log

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ -f exp/unsup_vae/log/report_${log_filename} ] && echo "! Exists exp/unsup_vae/log/report_${log_filename}" && tail -vn4 exp/unsup_vae/log/report_${log_filename} && exit 0;

if [ $stage -le 0 ]; then  
  # local/make_voxceleb1_sv.pl /lan/ibdata/SPEECH_DATABASE/voxceleb1 data
  echo ">> make filterbank features ---"
  for i in data/train data/$testdir ; do
    # steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$cpu_cmd" $i exp/make_fbank $featdir
    # mv $i/feats.scp $i/feats_fbank.scp
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$cpu_cmd" $i exp/make_mfcc $mfccdir
    # sid/compute_vad_decision.sh --nj 40 --cmd "$cpu_cmd" $i exp/make_vad $vaddir
    # cp $i/feats.scp $i/feats_mfcc.scp
    # cp $i/feats_fbank.scp $i/feats.scp
  done

  for name in train test; do
    utils/fix_data_dir.sh data/${name}
  done

  ## deprecated
  # compute-cmvn-stats scp:data/train/feats.scp data/train/cmvn_stats
fi

if [ $stage -le 1 ]; then
  # local/fix_feats_dur.sh --nj 40 --feats-dur 5 --raw-feats true data/train exp/feats_5s/train || exit 1;
  python local/unsup_vae_train.py --batch-size 64 \
                              --num-epochs 50 \
                              --gpu-idxs ${gpu_idxs} \
                              --resume-epoch ${resume_epoch} \
                              --log-step 500 \
                              --init-lr 1e-3 \
                              --end-lr 1e-4 \
                              --data-crop True \
                              --data-aug False \
                              --sort-utts True \
                              --frame-num-thresh 2000 \
                              --cluster-num ${cluster_num} \
                              --embed-dim ${embed_dim} \
                              --latent-dim ${latent_dim} \
                              --feat-dim ${feat_dim} \
                              --bnf-feat-dim ${bnf_feat_dim} \
                              --cmvn-apply-global ${cmvn_apply_global} \
                              --cmvn-norm-vars ${cmvn_norm_vars} \
                              --cmvn-reverse ${cmvn_reverse} \
                              --cmvn-stats-rxfilename ${cmvn_stats_rxfilename} \
                              --vad-label False \
                              --use-ivec False \
                              --train-data-dir data/train \
                              --log-filename simp_train_${log_filename}
fi

if [ $stage -le 2 ]; then
  if [ "$mode" != "debug" ]; then
    python local/unsup_compute_embeddings.py --batch-size 1 \
                                       --data-crop False \
                                       --data-aug False \
                                       --gpu-idxs ${gpu_idxs} \
                                       --frame-num-thresh 4000 \
                                       --cluster-num ${cluster_num} \
                                       --embed-dim ${embed_dim} \
                                       --latent-dim ${latent_dim} \
                                       --bnf-feat-dim ${bnf_feat_dim} \
                                       --feat-dim ${feat_dim} \
                                       --cmvn-apply-global ${cmvn_apply_global} \
                                       --cmvn-norm-vars ${cmvn_norm_vars} \
                                       --cmvn-reverse ${cmvn_reverse} \
                                       --cmvn-stats-rxfilename ${cmvn_stats_rxfilename} \
                                       --ipath2model ${path2model} \
                                       --log-filename "compute_embeddings_${log_filename}" \
                                       --use-ivec False \
                                       --vad-label False
  else
    echo "deprecated!"
    # # deprecated
    # python local/unsup_compute_posts.py --batch-size 1 \
    #                                     --data-crop False \
    #                                     --data-aug False \
    #                                     --gpu-idxs ${gpu_idxs} \
    #                                     --frame-num-thresh 4000 \
    #                                     --cluster-num ${cluster_num} \
    #                                     --embed-dim ${embed_dim} \
    #                                     --bnf-feat-dim ${bnf_feat_dim} \
    #                                     --latent-dim ${latent_dim} \
    #                                     --feat-dim ${feat_dim} \
    #                                     --cmvn-apply-global ${cmvn_apply_global} \
    #                                     --cmvn-norm-vars ${cmvn_norm_vars} \
    #                                     --cmvn-reverse ${cmvn_reverse} \
    #                                     --cmvn-stats-rxfilename data/train/cmvn_stats \
    #                                     --ipath2model ${path2model} \
    #                                     --log-filename "compute_posts_${log_filename}" \
    #                                     --use-ivec False \
    #                                     --vad-label False

    # python local/vector-mean.py \
    #     scp:exp/train_embed_vectors/embeddings.scp \
    #     exp/train_embed_vectors/mean.vec

    # python local/unsup_compute_feats_reconstr.py --batch-size 1 \
    #                                              --data-crop False \
    #                                              --data-aug False \
    #                                              --gpu-idxs ${gpu_idxs} \
    #                                              --frame-num-thresh 4000 \
    #                                              --cluster-num ${cluster_num} \
    #                                              --embed-dim ${embed_dim} \
    #                                              --bnf-feat-dim ${bnf_feat_dim} \
    #                                              --latent-dim ${latent_dim} \
    #                                              --feat-dim ${feat_dim} \
    #                                              --cmvn-apply-global ${cmvn_apply_global} \
    #                                              --cmvn-norm-vars ${cmvn_norm_vars} \
    #                                              --cmvn-reverse ${cmvn_reverse} \
    #                                              --cmvn-stats-rxfilename data/train/cmvn_stats \
    #                                              --ipath2model ${path2model} \
    #                                              --log-filename "compute_feats_reconstr_${log_filename}" \
    #                                              --use-ivec False \
    #                                              --vad-label False \
    #                                              --ipath2spkvector exp/train_embed_vectors/mean.vec


    exit 0;
  fi
fi

if [ $stage -le 3 ] && $nnscore ; then
  # python local/unsup_vae_finetune_pairclf.py --batch-size 64 \
  #                                            --gpu-idxs ${gpu_idxs} \
  #                                            --cluster-num ${cluster_num} \
  #                                            --embed-dim ${embed_dim} \
  #                                            --bnf-feat-dim ${bnf_feat_dim} \
  #                                            --feat-dim ${feat_dim} \
  #                                            --latent-dim ${latent_dim} \
  #                                            --resume-epoch 50 \
  #                                            --num-epochs 52 \
  #                                            --init-lr 1e-3 \
  #                                            --end-lr 1e-4 \
  #                                            --data-aug False \
  #                                            --sort-utts True \
  #                                            --frame-num-thresh 2000 \
  #                                            --log-step 500 \
  #                                            --cmvn-apply-global ${cmvn_apply_global} \
  #                                            --cmvn-norm-vars ${cmvn_norm_vars} \
  #                                            --cmvn-reverse ${cmvn_reverse} \
  #                                            --cmvn-stats-rxfilename data/train/cmvn_stats \
  #                                            --log-filename "fineTune_clf_${log_filename}" \
  #                                            --use-ivec False \
  #                                            --vad-label True

  # python local/unsup_embed_score.py --batch-size 64 \
  #                                   --gpu-idxs ${gpu_idxs} \
  #                                   --cluster-num ${cluster_num} \
  #                                   --embed-dim ${embed_dim} \
  #                                   --bnf-feat-dim ${bnf_feat_dim} \
  #                                   --latent-dim ${latent_dim} \
  #                                   --cmvn-apply-global ${cmvn_apply_global} \
  #                                   --cmvn-norm-vars ${cmvn_norm_vars} \
  #                                   --cmvn-reverse ${cmvn_reverse} \
  #                                   --cmvn-stats-rxfilename data/train/cmvn_stats \
  #                                   --ipath2model exp/unsup_vae/52.mdl \
  #                                   --log-filename "score_embed_${log_filename}" \
  #                                   --use-ivec False \
  #                                   --vad-label False

  # python local/unsup_embed_llk.py --batch-size 1 \
  #                                 --gpu-idxs ${gpu_idxs} \
  #                                 --cluster-num ${cluster_num} \
  #                                 --embed-dim ${embed_dim} \
  #                                 --bnf-feat-dim ${bnf_feat_dim} \
  #                                 --latent-dim ${latent_dim} \
  #                                 --cmvn-apply-global ${cmvn_apply_global} \
  #                                 --cmvn-norm-vars ${cmvn_norm_vars} \
  #                                 --cmvn-reverse ${cmvn_reverse} \
  #                                 --cmvn-stats-rxfilename data/train/cmvn_stats \
  #                                 --ipath2model exp/unsup_vae/50.mdl \
  #                                 --log-filename "score_embed_${log_filename}"
 
  # eer=$(paste data/voxceleb1_trials_sv exp/unsup_vae/log/nn_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  # echo "Using Unsup-NN, EER: ${eer}%"
  exit 0;
fi

if [ $stage -le 3 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $cpu_cmd exp/train_embed_vectors/log/compute_mean.log \
    ivector-mean scp:exp/train_embed_vectors/embeddings.scp \
    exp/train_embed_vectors/mean.vec || exit 1;

  lda_dim=150
  $cpu_cmd exp/train_embed_vectors/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/train_embed_vectors/embeddings.scp ark:- |" \
    ark:data/train/utt2spk exp/train_embed_vectors/transform.mat || exit 1;

  $cpu_cmd exp/train_embed_vectors/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/train_embed_vectors/embeddings.scp ark:- | transform-vec exp/train_embed_vectors/transform.mat ark:- ark:- |" \
    exp/train_embed_vectors/plda || exit 1;

fi

if [ $stage -le 4 ]; then
  # do cosine scoring
  $cpu_cmd exp/scores/log/cosine_scoring.log \
    cat $trials \| awk '{print $1" "$2}' \| \
    ivector-compute-dot-products - \
      "ark:ivector-subtract-global-mean exp/train_embed_vectors/mean.vec scp:exp/test_embed_vectors/embeddings.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean exp/train_embed_vectors/mean.vec scp:exp/test_embed_vectors/embeddings.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      exp/scores/cosine_scores || exit 1;

  eer=$(paste $trials exp/scores/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Cosine scoring, EER: ${eer}%"
  echo "Time: $(date)." >> exp/unsup_vae/log/eerScores_${log_filename}
  echo "Cosine scoring, EER: ${eer}%" >> exp/unsup_vae/log/eerScores_${log_filename}
fi

if [ $stage -le 5 ]; then

  # do lda + cosine scoring
  $cpu_cmd exp/scores/log/lda_scoring.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-subtract-global-mean exp/train_embed_vectors/mean.vec scp:exp/test_embed_vectors/embeddings.scp ark:- | ivector-transform exp/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/train_embed_vectors/mean.vec scp:exp/test_embed_vectors/embeddings.scp ark:- | ivector-transform exp/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/scores/lda_scores || exit 1;
    
  eer=$(paste $trials exp/scores/lda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "LDA + cosine scoring, EER: ${eer}%"
  echo "LDA + cosine scoring, EER: ${eer}%" >> exp/unsup_vae/log/eerScores_${log_filename}

fi

if [ $stage -le 6 ]; then
  # Get result using the out-of-domain PLDA model.
  # Note that this is not the correct way to do plda scoring.
  # However, the paper: "VoxCeleb: a large-scale speaker identification dataset" uses this approach.
  # So we just follow it to make a fair comparision.

  $cpu_cmd exp/train_embed_vectors/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean exp/train_embed_vectors/mean.vec scp:exp/train_embed_vectors/embeddings.scp ark:- | transform-vec exp/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/train_embed_vectors/plda || exit 1;

  $cpu_cmd exp/scores/log/plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
      "ivector-copy-plda --smoothing=0.0 exp/train_embed_vectors/plda - |" \
      "ark:ivector-subtract-global-mean exp/train_embed_vectors/mean.vec scp:exp/test_embed_vectors/embeddings.scp ark:- | transform-vec exp/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean exp/train_embed_vectors/mean.vec scp:exp/test_embed_vectors/embeddings.scp ark:- | transform-vec exp/train_embed_vectors/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$trials' | cut -d\  --fields=1,2 |" exp/scores/plda_scores || exit 1;

  eer=$(paste $trials exp/scores/plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: ${eer}%"
  echo "Using Out-of-Domain PLDA, EER: ${eer}%" >> exp/unsup_vae/log/eerScores_${log_filename}

fi

for i in simp_train compute_embeddings eerScores; do
  cat exp/unsup_vae/log/${i}_${log_filename}
done >> exp/unsup_vae/log/report_${log_filename}

