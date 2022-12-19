# HAKE-Action
HAKE-Action (TensorFlow) is a project to open the SOTA action understanding studies based on our [Human Activity Knowledge Engine](http://hake-mvig.cn/home/). It includes reproduced SOTA models and their HAKE-enhanced versions.
HAKE-Action is authored by [Yong-Lu Li](https://dirtyharrylyl.github.io/), Xinpeng Liu, Liang Xu, Cewu Lu. Currently, it is manintained by Yong-Lu Li, Xinpeng Liu and Liang Xu.

#### **News**: (2022.12.19) HAKE 2.0 is accepted by TPAMI!

(2022.11.19) We release the interactive object bounding boxes & classes in the interactions within AVA dataset (2.1 & 2.2)! [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA), [[Paper]](https://arxiv.org/abs/2211.07501). BTW, we also release a CLIP-based human body part states recognizer in [CLIP-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/CLIP-Activity2Vec)!

(2022.07.29) Our new work PartMap (ECCV'22) is released! [Paper](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness/blob/main), [Code](https://github.com/DirtyHarryLYL/HAKE-Action-Torch)

(2022.02.14) We release the human body part state labels based on AVA: [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA) and [HAKE 2.0 paper](https://arxiv.org/abs/2202.06851).

(2021.10.06) Our extended version of [SymNet](https://github.com/DirtyHarryLYL/SymNet) is accepted by TPAMI! Paper and code are coming soon.

(2021.2.7) Upgraded [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec) is released! Images/Videos --> human box + ID + skeleton + part states + action + representation. [[Description]](https://drive.google.com/file/d/1iZ57hKjus2lKbv1MAB-TLFrChSoWGD5e/view?usp=sharing)
<p align='center'>
    <img src="https://github.com/DirtyHarryLYL/HAKE-Action-Torch/blob/Activity2Vec/demo/a2v-demo.gif", height="400">
</p>

<!-- ## Full demo: [[YouTube]](https://t.co/hXiAYPXEuL?amp=1), [[bilibili]](https://www.bilibili.com/video/BV1s54y1Y76s) -->

(2021.1.15) Our extended version of [TIN (Transferable Interactiveness Network)](https://arxiv.org/abs/2101.10292) is accepted by TPAMI! New paper and code will be released soon.

(2020.10.27) The code of [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) ([Paper](https://arxiv.org/abs/2010.16219)) in NeurIPS'20 is released!

(2020.6.16) Our larger version [HAKE-Large](https://github.com/DirtyHarryLYL/HAKE#hake-large-for-instance-level-hoi-detection) (>120K images, activity and part state labels) is released!

**We released the HAKE-HICO (image-level part state labels upon HICO) and HAKE-HICO-DET (instance-level part state labels upon HICO-DET). The corresponding data can be found here: [HAKE Data](https://github.com/DirtyHarryLYL/HAKE).**

- Paper is [here](https://arxiv.org/abs/2004.00945). 
- More data and part states (e.g., upon AVA, more kinds of action categories, more rare actions...) are coming.
- We will keep updating HAKE-Action to include more SOTA models and their HAKE-enhanced versions.

## [Data Mode](https://github.com/DirtyHarryLYL/HAKE)
- **HAKE-HICO** (**PaStaNet\* mode** in [paper](https://arxiv.org/abs/2004.00945)): image-level, add the aggression of all part states in an image (belong to one or multiple active persons), compared with original [HICO](http://www-personal.umich.edu/~ywchao/hico/), the only additional labels are image-level human body part states.

- **HAKE-HICO-DET** (**PaStaNet\*** in [paper](https://arxiv.org/abs/2004.00945)): instance-level, add part states for each annotated persons of all images in [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/), the only additional labels are instance-level human body part states.

- **HAKE-Large** (**PaStaNet** in [paper](https://arxiv.org/abs/2004.00945)): contains more than 120K images, action labels and the corresponding part state labels. The images come from the existing action datasets and crowdsourcing. We mannully annotated all the active persons with our novel part-level semantics.

- **GT-HAKE** (**GT-PaStaNet\*** in [paper](https://arxiv.org/abs/2004.00945)): GT-HAKE-HICO and G-HAKE-HICO-DET. It means that we use the part state labels as the part state prediction. That is, we can **perfectly** estimate the body part states of a person. Then we use them to infer the instance activities. This mode can be seen as the **upper bound** of our HAKE-Action. From the results below we can find that, the upper bound is far beyond the SOTA performance. Thus, except for the current study on the conventional instance-level method, continue promoting **part-level** method based on HAKE would be a very promising direction.

## Notion
Activity2Vec and PaSta-R are our part state based modules, which operate action inference based on part semantics, different from previous instance semantics. For example, **Pairwise + HAKE-HICO pre-trained Activity2Vec + Linear PaSta-R** (the seventh row) achieves 45.9 mAP on HICO. More details can be found in our CVPR2020 paper: [PaStaNet: Toward Human Activity Knowledge Engine](https://arxiv.org/abs/2004.00945).

## Code
The two versions of HAKE-Action are relesased in two branches of this repo:
- [Image-level Action Recognition](https://github.com/DirtyHarryLYL/HAKE-Action/tree/Image-level-HAKE-Action)

- [Instance-level Action Detection](https://github.com/DirtyHarryLYL/HAKE-Action/tree/Instance-level-HAKE-Action)

## [Models on HICO](https://github.com/DirtyHarryLYL/HAKE-Action/tree/Image-level-HAKE-Action)
|Instance-level| +Activity2Vec | +PaSta-R | mAP | Few@1 | Few@5 | Few@10 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
[R\*CNN](https://arxiv.org/pdf/1505.01197.pdf) | - | - | 28.5 | -| - | -|
[Girdhar et.al.](https://arxiv.org/pdf/1711.01467.pdf) | -|-|34.6|-|-|-|
[Mallya et.al.](https://arxiv.org/pdf/1604.04808.pdf) | -|-|36.1|-|-|-|
[Pairwise](http://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshu_Fang_Pairwise_Body-Part_Attention_ECCV_2018_paper.pdf) | -|-|39.9 | 13.0 | 19.8 | 22.3|
-|HAKE-HICO|Linear | 44.5 | **26.9** | 30.0 | 30.7 |
Mallya et.al.|HAKE-HICO | Linear | 45.0 |26.5 |29.1 | 30.3 |
Pairwise|HAKE-HICO | Linear | **45.9** | 26.2 | 30.6 | 31.8 |
Pairwise|HAKE-HICO|MLP | 45.6 | 26.0 | **30.8** | **31.9** |
Pairwise|HAKE-HICO|GCN | 45.6 | 25.2 | 30.0 | 31.4 |
Pairwise|HAKE-HICO|Seq | **45.9** | 25.3 | 30.2 | 31.6 |
Pairwise|HAKE-HICO|Tree | 45.8 | 24.9 | 30.3 | 31.8 |
Pairwise|HAKE-Large | Linear |**46.3** | 24.7 | **31.8** | **33.1**|
Pairwise|HAKE-Large | Linear |**46.3** | 24.7 | **31.8** | **33.1**|
Pairwise|GT-HAKE-HICO|Linear | **65.6** | **47.5** | **55.4** | **56.6** |

## [Models on HICO-DET](https://github.com/DirtyHarryLYL/HAKE-Action/tree/Instance-level-HAKE-Action)

**Using Object Detections from [iCAN](https://github.com/vt-vl-lab/iCAN)**
|Instance-level| +Activity2Vec | +PaSta-R | Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[iCAN](https://arxiv.org/pdf/1808.10437.pdf) |-|-| 14.84|  10.45 | 16.15 | 16.26  | 11.33| 17.73 |
|[TIN](https://arxiv.org/pdf/1811.08264.pdf) |-|-| 17.03 | 13.42 | 18.11 | 19.17 | 15.51 | 20.26 |
|iCAN | HAKE-HICO-DET|Linear| 19.61 | 17.29 | 20.30 | 22.10 | 20.46 | 22.59 |
|TIN | HAKE-HICO-DET|Linear| 22.12 | 20.19 | 22.69 | 24.06 | 22.19 | 24.62 |
|TIN | HAKE-Large |Linear| **22.65** | **21.17** | **23.09** | **24.53** | **23.00** | **24.99** |
|TIN | GT-HAKE-HICO-DET |Linear|**34.86** | **42.83** | **32.48** | **35.59** | **42.94** | **33.40**| 

## [Models on AVA](https://github.com/DirtyHarryLYL/HAKE-Action/tree/Instance-level-HAKE-Action) (Frame-based)
|Method| +Activity2Vec | +PaSta-R|mAP |
|:---:|:---:|:---:|:---:|
|[AVA-TF-Baseline](http://research.google.com/ava/download.html)| -|-|11.4 |
|[LFB-Res-50-baseline](https://github.com/facebookresearch/video-long-term-feature-banks)| -|-|22.2 |
|[LFB-Res-101-baseline](https://github.com/facebookresearch/video-long-term-feature-banks)| -|-|23.3 |
|AVA-TF-Baeline | HAKE-Large|Linear| 15.6 |
|LFB-Res-50-baseline | HAKE-Large|Linear | 23.4 |
|LFB-Res-101-baseline | HAKE-Large|Linear | 24.3 |

## Models on V-COCO
|Method| +Activity2Vec | +PaSta-R|AP(role), Scenario 1 | AP(role), Scenario 2 |
|:---:|:---:|:---:|:---:|:---:|
|[iCAN](https://arxiv.org/pdf/1808.10437.pdf) |-|-| 45.3 | 52.4 |
|[TIN](https://arxiv.org/pdf/1811.08264.pdf)  |-|-| 47.8 | 54.2 |
|iCAN | HAKE-Large|Linear   | 49.2 | 55.6 |
|TIN |HAKE-Large|Linear    | **51.0** | **57.5** |

## Training Details
We first pre-train the Activity2Vec and PaSta-R with activities and PaSta labels.
Then we change the last FC in PaSta-R to fit the activity categories of the target dataset.
Finally, we freeze Activity2Vec and fine-tune PaSta-R on the train set of the target dataset.
Here, HAKE works like the ImageNet and Activity2Vec is used as a pre-trained knowledge engine to promote other tasks.

## Citation
If you find our work useful, please consider citing:
```
    @article{li2023hake,
     title={HAKE: A Knowledge Engine Foundation for Human Activity Understanding},
    author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Li, Yizhuo and Qiu, Zuoyu and Xu, Liang and Xu, Yue and Fang, Hao-Shu and Lu, Cewu},
    journal={TPAMI},
    year={2023}
    }
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
@inproceedings{li2019transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Zhou, Siyuan and Huang, Xijie and Xu, Liang and Ma, Ze and Fang, Hao-Shu and Wang, Yanfeng and Lu, Cewu},
  booktitle={CVPR},
  year={2019}
}
@inproceedings{lu2018beyond,
  title={Beyond holistic object recognition: Enriching image understanding with part states},
  author={Lu, Cewu and Su, Hao and Li, Yonglu and Lu, Yongyi and Yi, Li and Tang, Chi-Keung and Guibas, Leonidas J},
  booktitle={CVPR},
  year={2018}
}
```

## [HAKE](http://hake-mvig.cn/home/)
**HAKE**[[website]](http://hake-mvig.cn/home/) is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the action understanding performance on widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

HAKE-Action is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
