module unload cudnn/8.7.0.84-11.8
module unload cuda/11.8.0

#       lbatch -g 1 -t 24 \
#   -q gypsum-2080ti --name BN_${bs}_${LR}_111 \
#   --memory 50 -c 8 --cmd \
#   python main.py  --LR 1e-06  --use_data_aug \
#   --total_epoch 200 --bs 50 --mbs 25 --seed 111 &


#           lbatch -g 1 -t 24 \
#     -q gypsum-2080ti --name BN_${bs}_${LR}_111 \
#     --memory 50 -c 8 --cmd \
#     python main.py  --LR 1.0  --use_data_aug \
#     --total_epoch 200 --bs 50 --mbs 25 --seed 111 &


# for seed in 1 111 11111
for seed in 11 111 11111
do
# for LR in 1e-6 1e-5 1e-4 1e-3
for LR in 1e-7
# for LR in 1e-2 1e-1 1e0
do
for mbs in 1000
do
# for bs in 25 50 100 200 400 800
for bs in 1000 2000 5000 10000 25000
# for bs in 1600 3200
do

      lbatch -g 1 -t 24 \
        -q gpu-long --name ${bs}_${LR}_${seed} \
        --memory 50 -c 8 --cmd \
        python main.py --LR $LR  --use_data_aug \
        --total_epoch 400 --bs $bs --mbs $mbs --seed $seed --KMAX 1  &

      # lbatch -g 1 -t 24 \
      #   -q gpu --name ${bs}_${LR}_${seed} \
      #   --memory 50 -c 8 --cmd \
      #   python main.py --LR $LR  --use_data_aug \
      #   --total_epoch 200 --bs $bs --mbs $mbs --seed $seed &

      # lbatch -g 1 -t 24 \
      # -q gypsum-titanx --name BN_${bs}_${LR}_1 \
      # --memory 50 -c 8 --cmd \
      # python main.py --LR $LR  --use_data_aug \
      # --total_epoch 200 --bs $bs --mbs $mbs --seed 1 &

# TODO: rerun this
    #     lbatch -g 1 -t 24 \
    # -q gypsum-1080ti --name BN_${bs}_${LR}_111 \
    # --memory 50 -c 8 --cmd \
    # python main.py  --LR $LR  --use_data_aug \
    # --total_epoch 200 --bs $bs --mbs $mbs --seed 111 &

      #   lbatch -g 1 -t 24 \
      # -q gypsum-2080ti --name BN_${bs}_${LR}_11111 \
      # --memory 50 -c 8 --cmd \
      # python main.py  --LR $LR  --use_data_aug \
      # --total_epoch 200 --bs $bs --mbs $mbs --seed 11111 &
done
done
done
done


wait
