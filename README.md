# Instance-level-HAKE-Action
Code for our CVPR2020 paper *"PaStaNet: Toward Human Activity Knowledge Engine"*.

Link: [[arXiv]](https://arxiv.org/abs/2004.00945)

## Results on HICO-DET, AVA and VCOCO

### [Data Mode](https://github.com/DirtyHarryLYL/HAKE)
- **HAKE-HICO-DET** (**PaStaNet\*** in [paper](https://arxiv.org/abs/2004.00945)):instance-level, add part states for each annotated persons of all images in [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/), the only additional labels are instance-level human body part states.

- **HAKE-Large** (**PaStaNet** in [paper](https://arxiv.org/abs/2004.00945)): contains more than 120K images, action labels and the corresponding part state labels. The images come from the existing action datasets and crowdsourcing. We mannully annotated all the active persons with our novel part-level semantics.

- **GT-HAKE-HICO-DET** (**GT-PaStaNet\*** in [paper](https://arxiv.org/abs/2004.00945)): means that if we use the part state labels as the part stat prediction. That is, we can **perfectly** estimate the body part states of a person. Then use then to infer the instance activities. This mode can be seen as the **upper bound** of our HAKE-Action. From the results below we can find that, the upper bound is far beyond the SOTA performance. Thus, except for the current study on the conventional instance-level method, continue promoting **part-level** method based on HAKE would be a very promising direction.

**Our results on HICO-DET dataset, using object detections from iCAN**

|Method| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|iCAN           | 19.61 | 17.29 | 20.30 | 22.10 | 20.46 | 22.59 |
|TIN            | 17.03 | 13.42 | 18.11 | 19.17 | 15.51 | 20.26 |
|TIN + HAKE-HICO-DET| 22.12 | 20.19 | 22.69 | 24.06 | 22.19 | 24.62 |
|TIN + HAKE-Large | 22.65 | 21.17 | 23.09 | 24.53 | 23.00 | 24.99 |

**Our results (frame-based w/o temporal supervision) on AVA dataset, using detections from LFB**

|Method| mAP |
|:---:|:---:|
|AVA-TF-Baseline                 | 11.4 |
|LFB-Res-50-baseline             | 22.2 |
|LFB-Res-101-baseline            | 23.3 |
|AVA-TF-Baeline + HAKE-Large       | 15.6 |
|LFB-Res-50-baseline + HAKE-Large  | 23.4 |
|LFB-Res-101-baseline + HAKE-Large | 24.3 |

**Our results on VCOCO dataset, using object detections from iCAN**

During Activity2Vec and PaSta-R pre-training, the V-COCO data in HAKE are all exlcuded to avoid data polluation.
|Method| AP(role), Scenario 1 | AP(role), Scenario 2 |
|:---:|:---:|:---:|
|iCAN                     | 45.3 | 52.4 |
|TIN                      | 47.8 | 54.2 |
|iCAN + HAKE-Large   | 49.2 | 55.6 |
|TIN + HAKE-Large    | 51.0 | 57.5 |

## Getting Started

### Installation

1.Clone this repository.

```
git clone -b Instance-level-HAKE-Action https://github.com/DirtyHarryLYL/HAKE-Action.git
```

2.Download **dataset** and **pre-trained weights**. The detection results (person and object boudning boxes) are collected from [iCAN](http://chengao.vision/iCAN/).) And the part bounding boxes have been attached to the detection results. We show how to generate part bounding boxes with a toy example by [script/part_box_generation.py](https://github.com/DirtyHarryLYL/HAKE-Action/blob/Instance-level-HAKE-Action/script/part_box_generation.py).

```
chmod +x ./script/Dataset_download.sh 
./script/Dataset_download.sh
```

3.Install Python dependencies.

```
pip install -r requirements.txt
```

### Training

1.Pretrain Activity2Vec (PaStaNet* is HAKE-HICO-DET, PaStaNet is HAKE-Large)

```
python tools/Train_pasta_HICO_DET.py --data <0 for PaStaNet*, 1 for PaStaNet> --init_weight 1 --train_module 2 --num_iteration 2000000 --model <your model name>
```

2.Finetune on HICO-DET

```
python tools/Train_pasta_HICO_DET.py --data 0 --init_weight 2 --train_module 1 --num_iteration 2000000 --model <your model name>
```

3.Finetune on AVA 

```
python tools/Train_pasta_AVA.py --init_weight 3 --num_iteration 2000000 --model <your model name>
```


### Testing

1.Test on HICO-DET

```
python tools/Test_pasta_HICO_DET.py --iteration 2000000 --model <pasta_HICO-DET for PaStaNet*, pasta_full for PaStaNet>
```

2. Test on AVA
```
python tools/Test_pasta_HICO_DET.py --iteration 1100000 --model pasta_AVA_transfer
```


### Generate detection file

1. For HICO-DET

```
cd ./-Results/
python Generate_detection.py --model <pasta_HICO-DET for PaStaNet*, pasta_full for PaStaNet>
```

### Evaluation

1. For HICO-DET, we re-implemented the evaluation code in Python.
```
cd ./-Results/
python Evaluate_HICO_DET.py --file <detection file to be evaluated>
```

2. For AVA, we use the official evaluation code of AVA dataset.

### Q&A

Q: Are all the three branches in PaStaNet (PaSta branch, attention branch, language branch) necessary?

A: To get the best performance, they are all necessary. But the three branches in our model can be trained separately or jointly, thus it's easy to combine any of our branches with other state-of-the-art models to exploit the advantage of different branches in PaStaNet. For example, the language branch improves the upper bound significantly, while the attention branch plays an important role in joint training and improving the PaSta classification.

Q: Where does the `P_H, P_O, P_sp` in HICO-DET evaluation come from?

A: They are extracted from [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network). Feel free to replace them with the output of other models (with same input boxes), which would bring higher performances. We are also working on the combination of HAKE and recent sota models in new framework (faster training and higher perfromance). Once finished, we will release them in this repo.

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

## TODOS
- [x] Instance-level label results on HICO-DET and AVA
- [x] Instance-level code and models
- [x] HAKE-Large data, code and models
- [ ] HAKE-AVA data and code (video-based)
- [ ] Demo to process your own images, including detection, pose estimation, part box generation and HAKE-based HOI detection.

## Acknowledgement

Some of the codes are built upon [Interactiveness](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network) and [iCAN](https://github.com/vt-vl-lab/iCAN). 

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
