EXP_DIR=output
mkdir -p ${EXP_DIR}/log

now=$(date +"%Y%m%d_%H%M%S")
python scripts/train.py --config=depth_guided_config \
  2>&1 | tee output/log/$now.log