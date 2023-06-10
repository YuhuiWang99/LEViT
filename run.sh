## market 1501
# nohup python -u train.py --model dbn --net small --img-height 256 --img-width 128 --batch-size 64 --lr 1.5e-1 --dataset market1501 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &
# nohup python -u train.py --model dbn --net small --img-height 384 --img-width 128 --batch-size 64 --lr 1.5e-1 --dataset market1501 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &
# nohup python -u train.py --model dbn --net large --img-height 384 --img-width 128 --batch-size 64 --lr 1.5e-1 --dataset market1501 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &

# ## msmt
# nohup python -u train.py --model dbn --net small --freeze stem --img-height 256 --img-width 128 --batch-size 96 --lr 2.0e-1 --dataset msmt17 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &
# nohup python -u train.py --model dbn --net small --freeze stem --img-height 384 --img-width 128 --batch-size 96 --lr 2.0e-1 --dataset msmt17 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &
# nohup python -u train.py --model dbn --net large --freeze stem --img-height 384 --img-width 128 --batch-size 96 --lr 2.0e-1 --dataset msmt17 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &

# ## VeRi-776
# nohup python -u train.py --model dbn --net small --freeze stem --img-height 256 --img-width 256 --batch-size 96 --lr 2.0e-1 --dataset veri776 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &
# nohup python -u train.py --model dbn --net large --freeze stem,layer1 --img-height 256 --img-width 256 --batch-size 64 --lr 1.5e-1 --dataset veri776 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &

# ## VehicleID
# nohup python -u train.py --model dbn --net small --img-height 256 --img-width 256 --batch-size 320 --lr 2.5e-1 --dataset vehicleid --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &
# nohup python -u train.py --model dbn --net large --img-height 256 --img-width 256 --batch-size 320 --lr 2.5e-1 --dataset vehicleid --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25 &
