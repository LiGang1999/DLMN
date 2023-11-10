# When Masked Image Modeling Meets Source-free Unsupervised Domain Adaptation: Dual-Level Masked Network for Semantic Segmentation (Accepted at MM 2023)

This is the official implementation of the method in our paper [When Masked Image Modeling Meets Source-free Unsupervised Domain Adaptation: Dual-Level Masked Network for Semantic Segmentation](https://doi.org/10.1145/3581783.3612521).

## Setup

### Installation

```
pip install -r requirements.txt
```

### Data Preparation

Download [Cityscapes](https://www.cityscapes-dataset.com/).

### Model Preparation
Download model weights from [Official GtA adapted model from GTA5 to Cityscapes](https://drive.google.com/drive/folders/1MYZq6DPK6xemSM1yBz3UkwRvBW9rot5q?usp=sharing) and copy it to `checkpoints`.

Download model weights from [Reproduced GtA adapted model from SYNTHIA to Cityscapes](https://drive.google.com/drive/folders/1U0jqzluki2GcEeobHuX79_3D_S28LbF9?usp=drive_link) and copy it to `checkpoints`.

## Training

```
bash command_DLMN.sh
```

## Evaluation

```
bash mst_eval.sh
```

## Citation

If you find this repository useful in your research, please consider citing:

```
@inproceedings{li2023masked,
  title={When Masked Image Modeling Meets Source-free Unsupervised Domain Adaptation: Dual-Level Masked Network for Semantic Segmentation},
  author={Li, Gang and Ma, Xianzheng and Wang, Zhao and Li, Hao and Zhang, Qifei and Wu, Chao},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7638--7647},
  year={2023}
}
```

## Acknowledgments

Our code is heavily borrowed from [GtA](https://sites.google.com/view/sfdaseg).

We also thank Lukas Hoyer for his code in the repository [MIC](https://github.com/lhoyer/MIC).
