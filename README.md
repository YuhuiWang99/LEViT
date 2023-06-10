# LEViT
LEViT: Locally Enhanced Vision Transformer for Efficient Object Re-identification.

## Pipeline 
![framework](figs/framework.jpg)

## Prepare Market1501 dataset
1. Download the person datasets [Market1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) to "./dataset" folder

Then unzip them and rename them under the directory like
```
dataset
├── Market-1501-v15.09.15
    └── bounding_box_test
    └── bounding_box_train
    └── gt_bbox
    └── gt_query
    └── query
    └── readme.txt
```

2. Generate the "train.txt" file
## Train with LEViT
```bash
python -u train.py --model dbn --net small --img-height 256 --img-width 128 --batch-size 64 --lr 1.5e-1 --dataset market1501 --gpus 0,1 --swa True --swa-ratio 0.80 --swa-extra 25
```

## Contact
If you have any questions, please contact us by email(wangyh99@stu.xjtu.edu.cn).

