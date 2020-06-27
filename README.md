# Image-level HAKE-Action
Our architecture for image-level experiment is based on [Caffe](https://github.com/BVLC/caffe).

For details of the network design, please refer to our paper **PaStaNet: Toward Human Activity Knowledge Engine**.

The [docker version](https://hub.docker.com/r/liangxuy/hoi-caffe) is also released. More details please refer to [INSTALL.md](./INSTALL.md)

## Results on HICO

### Our results on HICO dataset:
|Method| mAP | Few@1 | Few@5 | Few@10 | Result\_url |
|:---:|:---:|:---:|:---:|:---:|:---:|
Pairwise+PaStaNet\*-Linear | **45.9** | 26.2 | 30.6 | 31.8 | [link](https://drive.google.com/open?id=1zsUWiQStN-T992ZurDrnsc1ZLqMkGBrX) |
Pairwise+PaStaNet\*-MLP | 45.6 | 26.0 | **30.8** | **31.9** | [link](https://drive.google.com/open?id=1fNSQw0V8duuW-EDofF5bzTQr2Cigj7w5) |
Pairwise+PaStaNet\*-GCN | 45.6 | 25.2 | 30.0 | 31.4 | [link](https://drive.google.com/open?id=1DbNXSYutczmd2q-EGxU8MOHT2pQQ4BAi) |
Pairwise+PaStaNet\*-Seq | **45.9** | 25.3 | 30.2 | 31.6 | [link](https://drive.google.com/open?id=1akhOY89RiQbTtYr2vpaiUaxu_N_Gv6sW) |
Pairwise+PaStaNet\*-Tree | 45.8 | 24.9 | 30.3 | 31.8 | [link](https://drive.google.com/open?id=1D2C68lJ_cnVFADqiozcUd4nid0JxXVTH) |
PaStaNet\*-Linear | 44.5 | **26.9** | 30.0 | 30.7 | [link](https://drive.google.com/open?id=1PoYc0AhXeLlKowSzSPXlywS9St00mIDY) |
PaStaNet+GT-PaStaNet\*-Linear | 65.6 | 47.5 | 55.4 | 56.6 | [link](https://drive.google.com/open?id=1_r4FM782pt-ydkMZwHfDSn_9ss8r6V4-) |
PairWise+PaStaNet-Linear | **46.3** | 24.7 | **31.8** | **33.1** | [link](https://drive.google.com/open?id=1LNc08IlZWKB-IE1kWbL1sBx5h0YzLXmc) |

## Getting Started
### 1. Installation
Please see installation instructions for Image-level HAKE-Action(Caffe) in [INSTALL.md](./INSTALL.md).

### 2. Preparing Data
2.1. Use the following commands to download imageset of HICO to folder```data/hico```.

```
cd data/hico
wget -N http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz
tar -xvzf hico_20160224_det.tar.gz -C ./JPEGImages
```
Or download the images from this [Google Drive](https://drive.google.com/file/d/1Qc3SOXdjzVUd1LDuG0ZS7DsE4Kb0FYVr/view?usp=sharing).

2.2. Download the image-level part-state labels. Use the following commands to download the labels from [Google Drive](https://drive.google.com/open?id=188YsrcvGl3ead4qf4deZzDNIKg7sk5uB).

```
python scripts/Download_data.py 188YsrcvGl3ead4qf4deZzDNIKg7sk5uB data/hico/Annotations.tar.gz
cd data/hico
tar -xvzf Annotations.tar.gz
rm Annotations.tar.gz
```

2.3. Download the pretrained model from the ```Model Zoo``` and put the models to ```snaps``` folder.

```
python scripts/Download_data.py 1iZIE1rNDTNSaGQZcCTAyz1cJnyc4ycA8 snaps/pretrain_model.caffemodel
```

2.4. Download the detected boxes from [Google Drive](https://drive.google.com/open?id=1sYNTL2YVWLciHb3JwILKpGTylMs7gQ15) to ```data/hico```.

```
cd data/hico
python scripts/Download_data.py 1sYNTL2YVWLciHb3JwILKpGTylMs7gQ15 ./boxes.tar.gz
tar -xvzf boxes.tar.gz
```

2.5. We provide the code of generating human part boxes in [part_box_processing.py](scripts/part_box_processing.py) for your reference.

## Evaluate

1. Download the evaluation code. You can use the evaluation code from [this repo](https://github.com/ywchao/hico_benchmark), or just download the code from [Google drive](https://drive.google.com/open?id=1mvXAtCe0Yc7JUQXCu3D_wpWt7r048lGc).

```
python scripts/Download_data.py 1mvXAtCe0Yc7JUQXCu3D_wpWt7r048lGc ./hico_benchmark
```

2. Download the provided result file

3. Use the following commands to evaluate

```
cp The_result_file hico_benchmark/data/test-result.csv
cd hico_benchmark
matlab -nodesktop -nodisplay
eval_default_run
```

## Test
You can use the following scripts to test the models. ```pasta-mode``` can be ```linear/mlp/tree/gcn/seq```.

```
python scripts/test.py --pasta-mode linear
```

For HAKE-Large data, you can add parameter ```--data large``` as:

```
python scripts/test.py --pasta-mode linear --data large
```


## Train
You can use the following scripts to train the models.

```
python scripts/train.py --pasta-mode linear
```

For HAKE-Large data, you can add parameter ```--data large``` as:

```
python scripts/train.py --pasta-mode linear --data large
```

You can paste the code of train_xxx.prototxt in [models](./models) to [http://ethereon.github.io/netscope/#/editor](http://ethereon.github.io/netscope/#/editor) to visualize the networks.


## Model Zoo
| Method | Model\_url |
|:---:|:---:|
PaStaNet\*-Linear | [link](https://drive.google.com/open?id=1d3LkrJQK62xl6jspquiXeyv-WgN6p156) |
PaStaNet\*-MLP | [link](https://drive.google.com/open?id=1sD_OLwM6eRfkzcrWuPmSGdC7eVa3SNm1) |
PaStaNet\*-GCN | [link](https://drive.google.com/open?id=1c0ZN4lKeOU73sSsiXSP4luUtUBFbDpvX) |
PaStaNet\*-Seq | [link](https://drive.google.com/open?id=1N__5ATxTdlSbM4uNp9ubAclUPBj1djQG) |
PaStaNet\*-Tree | [link](https://drive.google.com/open?id=1PYQWb3HrTAtDiGBzDLSGcG1BcGyQ89HT) |
10v-attention | [link](https://drive.google.com/open?id=1bmd5wiaggNYY4LzDn-39WAsrwt_Vbapr) |
Language-model | [link](https://drive.google.com/open?id=1vuFyWWvIl2YV7pY2wYZd0gVD5Z_5uHvr) |
Pretrain-model | [link](https://drive.google.com/open?id=1iZIE1rNDTNSaGQZcCTAyz1cJnyc4ycA8) |


## Citation
If you find this work useful, please consider citing:
```
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

## Acknowledgement

Some of the codes are built upon [Pairwise Body-Part Attention for Recognizing Human-Object Interactions](http://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshu_Fang_Pairwise_Body-Part_Attention_ECCV_2018_paper.pdf). Thanks them for their great work! 

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
