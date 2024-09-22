# for mbs in 10 20 40 100 200 400 800
# do
# python test_time.py --use_bi_adam --mbs $mbs --bs 800
# done

# for mbs in 10 25 50 100 200 400
# do
# python test_time.py --use_bi_adam --mbs $mbs --bs 400
# done

for mbs in 10 25 50 100 200
do
python test_time.py --use_bi_adam --mbs $mbs --bs 200
done

# for mbs in 10 25 50 100 200 400 800
# do
# python test_time.py --use_bi_adam --mbs $mbs --bs 800
# done

# for mbs in 10 25 50 100
# do
# python test_time.py --use_bi_adam --mbs $mbs --bs 100
# done

# for mbs in 10 25 50
# do
# python test_time.py --use_bi_adam --mbs $mbs --bs 50
# done


# for mbs in 10 20 40 100 200 400 800
# do
# python test_time.py --use_bi_adam --mbs 800 --bs 100
# done