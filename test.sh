mkdir -p logs
TAG=_2e9 ./sweep_walk_len.sh --dataset lanl14argus --device 2 --pretrain-tag _wl4_2e9 \
  > logs/2e9_static.log 2>&1 &
WLS="1 2 3 4" TAG=_2e9 ./sweep_walk_len.sh --dataset lanl14argus --device 3 --trw --pretrain-tag _wl4_2e9 \
  > logs/2e9_temporal.log 2>&1 &
wait