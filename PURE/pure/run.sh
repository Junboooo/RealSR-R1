for i in {0..7}
do
  export CUDA_VISIBLE_DEVICES=${i}
  python -u /PURE-main/PURE-main/pure/pre_tokenize/pre_tokenize.py \
  --splits=8 \
  --rank=${i} \
  --in_filename /data/training_for_ar_LSFF/prompt_final_pipline.json \
  --out_dir /data/training_for_ar_LSFF/token_pure_3out \
  --target_size 512 &> ${i}.log &
done



python /PURE-main/PURE-main/pure/pre_tokenize/pre_tokenize.py --in_filename /data/training_for_ar_LSFF/prompt_final_pipline.json --out_dir /data/training_for_ar_LSFF/token_pure_3out  --target_size 512