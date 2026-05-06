# Dataset Setup

## POPE Benchmark

Download POPE annotation files from the official repository:

```bash
mkdir -p data/pope
cd data/pope

# Download all three splits
wget https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_random.json
wget https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_popular.json
wget https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json
```

## MS-COCO val2014 Images

```bash
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```

Expected structure:
```
data/
├── pope/
│   ├── coco_pope_random.json
│   ├── coco_pope_popular.json
│   └── coco_pope_adversarial.json
└── coco/
    └── val2014/
        ├── COCO_val2014_000000000042.jpg
        └── ...
```
