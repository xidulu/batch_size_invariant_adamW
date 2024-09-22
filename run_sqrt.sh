module unload cudnn/8.7.0.84-11.8
module unload cuda/11.8.0


partition_list=("gypsum-titanx" "gypsum-titanx" "gypsum-1080ti")
seed_list=(11 111 11111)

# for seed in 1 111 11111
for i in "${!seed_list[@]}"
do
# for LR in 1e-6 1e-5 1e-4 1e-3 1e-2
for LR in 1e-6 1e-5 1e-4 1e-3
# for LR in 1e-7
# for LR in 1e-2 1e-1 1e0
do
for mbs in 1000
do
for bs in 1000 2000 5000 10000
do
    seed=${seed_list[i]}
    partition=${partition_list[i]}

    lbatch -g 1 -t 24 \
    -q $partition --name Adam_BN_${bs}_${LR}_11 \
    --memory 50 -c 8 --cmd \
    python main.py  --LR $LR  --use_data_aug  \
    --total_epoch 400 --bs $bs --mbs $mbs --seed $seed --use_std_adam --KMAX 1 --use_grad_accum  &

done
done
done
done


wait
