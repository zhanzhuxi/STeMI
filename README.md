# STeMI

![generation_framework](./images/framework.png)

## Spatial-Temporal Multi-scale Interaction for Few-Shot Video Summarization

### Dependencies
This codes requires the following:
- Python 3.6
- Pytorch 1.13.1
- h5py 3.9.0
- ortools 8.0.8283
- scipy 1.10.1

### Datasets

Download the dataset (TVSum/SumMe) from [datasets](https://drive.google.com/drive/folders/1HAkqtixx1xz2bu2h654CNXEN36wTXIgr?usp=sharing), and save the corresponding data into the directory `./datasets/`.

### Train and Test

Run train.py by `python train.py --model-dir models/tvsum --splits splits/tvsum_few_shot.yml --temporal_scales 4 --spatial_scales 6` 

or

by `python train.py --model-dir models/summe --splits splits/summe_few_shot.yml --temporal_scales 4 --spatial_scales 2`
