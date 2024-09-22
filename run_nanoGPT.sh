module unload cudnn/8.7.0.84-11.8
module unload cuda/11.8.0


# for seed in 1 111 11111
for seed in 1 111 11111
# for seed in 11
do 
for LR in 1e-6 1e-5 1e-4 1e-3 1e-2
# for LR in 1e-2 1e-1 1e0
do
for mbs in 64
do
# for bs in 25 50 100 200 400 800
for bs in 64 128 256 512
do

      lbatch -g 1 -t 24 \
        -q gpu-long --name bi_adam_vit_${bs}_${LR}_${seed} \
        --memory 50 -c 8 --cmd \
        python nano_gpt_main.py --LR $LR --wd 0.01  \
        --total_epoch 100 --bs $bs --mbs $mbs --seed $seed --KMAX 1 &

      lbatch -g 1 -t 24 \
        -q gpu-long --name vit_${bs}_${LR}_${seed} \
        --memory 50 -c 8 --cmd \
        python nano_gpt_main.py --LR $LR --wd 0.01  \
        --total_epoch 100 --bs $bs --mbs $mbs --seed $seed --use_std_adam --KMAX 1  &

done
done
done
done


wait
