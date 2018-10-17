#!/bin/bash

# Copyright 2016  ASLP (Author: zhangbinbin)
# Apache 2.0

stage=5
feat_dir=data/subset_of_data6400
global_cmvn=data/subset_of_data6400/train_tr/global-cmvn
gmmdir=exp/tri2b
ali=exp/subset_of_data6400_ali
skip_width=0
gpu_id=5
lr=0.00001
dir=exp/135-4dfsmn_bmuf-lr$lr
num_cv_utt=5000
#graph=graph_000_009_kni_p1e8_3gram
graph=graph
sync_period=160000

echo "$0 $@"  # Print the command line for logging
[ -f cmd.sh ] && . ./cmd.sh;
[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

# Making features, this script will gen corresponding feat and dir
# eg data-fbank/train data-fbank/train_tr90 data-fbank/train_cv10 data-fbank/test
if [ $stage -le 0 ]; then
    echo "Extracting feats & Create tr cv set"
    # aslp_scripts/make_feats.sh --feat-type "fbank" data/train $feat_dir
    # Split tr & cv
    utils/shuffle_list.pl $feat_dir/train/feats.scp > $feat_dir/train/random.scp
    head -n$num_cv_utt $feat_dir/train/random.scp | sort > $feat_dir/train/cv.scp
    total=$(cat $feat_dir/train/random.scp | wc -l)
    left=$[$total-$num_cv_utt]
    tail -n$left $feat_dir/train/random.scp | sort > $feat_dir/train/tr.scp
    utils/subset_data_dir.sh --utt-list $feat_dir/train/tr.scp \
        $feat_dir/train $feat_dir/train_tr
    utils/subset_data_dir.sh --utt-list $feat_dir/train/cv.scp \
        $feat_dir/train $feat_dir/train_cv
    compute-cmvn-stats scp:$feat_dir/train_tr/feats.scp $global_cmvn
	# aslp_scripts/make_feats.sh --feat-type "fbank" data/test $feat_dir
fi

# Prepare feature and alignment config file for nn training
# This script will make $dir/train.conf automaticlly
if [ $stage -le 1 ]; then
    echo "Preparing alignment and feats"
    aslp_scripts/aslp_nnet/prepare_feats_ali_parallel.sh \
        --cmvn_opts "--norm-means=true --norm-vars=true" \
        --global-cmvn-file "$global_cmvn" \
        --splice_opts "--left-context=1 --right-context=1" \
        --num-worker 4 \
	    $feat_dir/train_tr $feat_dir/train_cv data/lang $ali $ali $dir || exit 1;
fi

# Get feats_tr feats_cv labels_tr labels_cv 
[ ! -f $dir/train.parallel.conf ] && \
    echo "$dir/train.parallel.conf(config file for nn training): no such file" && exit 1 
source $dir/train.parallel.conf

# Prepare fsmn init nnet
if [ $stage -le 2 ]; then
    echo "Pretraining nnet"
    num_feat=$(feat-to-dim "$feats_tr" -) 
    num_tgt=$(hmm-info --print-args=false $ali/final.mdl | grep pdfs | awk '{ print $NF }')
    echo $num_feat $num_tgt

# Init nnet.proto with 2 lstm layers
cat > $dir/nnet.proto <<EOF
<NnetProto>
<AffineTransform> <InputDim> $num_feat <OutputDim> 2048 <MaxNorm> 0.000000 <Xavier> 1
<ReLU> <InputDim> 2048 <OutputDim> 2048
<LinearTransform> <InputDim> 2048 <OutputDim> 512 <ParamStddev> 0.010000 <Xavier> 1
<Fsmn> <InputDim> 512 <OutputDim> 512  <LOrder> 20 <ROrder> 20 <LStride> 2 <RStride> 2
<DeepFsmn> <InputDim> 512 <OutputDim> 512 <HidSize> 2048  <LOrder> 20 <ROrder> 20 <LStride> 2 <RStride> 2
<DeepFsmn> <InputDim> 512 <OutputDim> 512 <HidSize> 2048  <LOrder> 20 <ROrder> 20 <LStride> 2 <RStride> 2
<DeepFsmn> <InputDim> 512 <OutputDim> 512 <HidSize> 2048  <LOrder> 20 <ROrder> 20 <LStride> 2 <RStride> 2
<AffineTransform> <InputDim> 512 <OutputDim> 2048 <Xavier> 1
<ReLU> <InputDim> 2048 <OutputDim> 2048
<AffineTransform> <InputDim> 2048 <OutputDim> 2048 <Xavier> 1
<ReLU> <InputDim> 2048 <OutputDim> 2048
<LinearTransform> <InputDim> 2048 <OutputDim> 512 <Xavier> 1
<AffineTransform> <InputDim> 512 <OutputDim> $num_tgt <Xavier> 1
<Softmax> <InputDim> $num_tgt <OutputDim> $num_tgt
</NnetProto>
EOF

fi

if [ $stage -le 3 ]; then
    # init model
	nnet_init=$dir/nnet/train.nnet.init
	aslp-nnet-init $dir/nnet.proto $nnet_init
	# warm start
	single_init=$dir/nnet/train.single.nnet.final
    
	aslp_scripts/aslp_nnet/train_scheduler.sh --train-tool "aslp-nnet-train-frame" \
        --learn-rate $lr \
        --momentum 0.9 \
        --max-iters 20 \
        --minibatch_size 1024 \
		--train-tool-opts "--gpu-id=$gpu_id --report-period=1000 --randomize=false" \
        $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
	#cp $dir/final.nnet $single_init
	#rm $dir/final.nnet
fi
# 
# # Train nnet(dnn, cnn, lstm)
# if [ $stage -le 4 ]; then
#     echo "Training nnet"
# 	single_final=$dir/nnet/train.single.nnet.final
# 	parallel_init=$dir/nnet/train.parallel.nnet.init
#     cp $single_final $parallel_init
# 	aslp_scripts/aslp_nnet/train_scheduler_4workers.sh --train-type "bmuf" \
#         --learn-rate 0.0000008 --momentum 0.9 \
#         --minibatch_size 1024 \
# 		--gpu-num 4 --gpu-id $gpu_id \
# 		--max-iters 40 \
#         --worker-tool-opts "--bmuf-learn-rate=1.0 --bmuf-momentum=0.75 --sync-period=$sync_period" \
# 		--train-tool "aslp-nnet-train-frame" \
# 		--worker-tool "aslp-nnet-train-frame-worker" \
# 		--train-tool-opts "--report-period=1000 --randomize=false" \
#         $parallel_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
# fi

# Decoding 
if [ $stage -le 5 ]; then
    for x in test3000_fbank test3000_noise_fbank; do
		aslp_scripts/aslp_nnet/decode.sh --nj 5 --num-threads 3 \
        	--cmd "$decode_cmd" --acwt 0.0666667 \
        	--nnet-forward-opts "--no-softmax=false --apply-log=true --skip-width=$skip_width" \
        	--forward-tool "aslp-nnet-forward" \
        	$gmmdir/$graph $feat_dir/${x} $dir/decode_${x}_${graph} || exit 1;
    	aslp_scripts/score_basic.sh --cmd "$decode_cmd" $feat_dir/${x} \
        	$gmmdir/$graph $dir/decode_${x}_${graph} || exit 1;
	done
fi

