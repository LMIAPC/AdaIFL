# AdaIFL: Adaptive Image Forgery Localization via a Dynamic and Importance-aware Transformer Network
--------

This repo contains an official implementation of our paper: [AdaIFL: Adaptive Image Forgery Localization via a Dynamic and Importance-aware Transformer Network.](https://link.springer.com/chapter/10.1007/978-3-031-72775-7_27)

## Overview
The rapid development of image processing and manipulation techniques poses unprecedented challenges in multimedia forensics, especially in Image Forgery Localization (IFL). This paper addresses two key challenges in IFL: (1) Various forgery techniques leave distinct forensic traces. However, existing models overlook variations among forgery patterns. The diversity of forgery techniques makes it challenging for a single static detection method and network structure to be universally applicable. To address this, we propose AdaIFL, a dynamic IFL framework that customizes various expert groups for different network components, constructing multiple distinct feature subspaces. By leveraging adaptively activated experts, AdaIFL can capture discriminative features associated with forgery patterns, enhancing the model's generalization ability. (2) Many forensic traces and artifacts are located at the boundaries of the forged region. Existing models either ignore the differences in discriminative information or use edge supervision loss to force the model to focus on the region boundaries. This hard-constrained approach is prone to attention bias, causing the model to be overly sensitive to image edges or fail to finely capture all forensic traces. In this paper, we propose a feature importance-aware attention, a flexible approach that adaptively perceives the importance of different regions and aggregates region features into variable-length tokens, directing the model's attention towards more discriminative and informative regions. Extensive experiments on benchmark datasets demonstrate that AdaIFL outperforms state-of-the-art image forgery localization methods.

## Environment

Python 3.8

PyTorch 2.0.1

## Installation
```
pip install -r requirements.txt
```

## Quick Start
To test the AdaIFL, simply run ```test.py```

```
python test.py --image image_path --model model_path --output output_path
```

## Citation
If AdaIFL helps your research or work, please cite our paper.
```
@inproceedings{li2025adaifl,
  title={AdaIFL: Adaptive Image Forgery Localization via a Dynamic and Importance-Aware Transformer Network},
  author={Li, Yuxi and Cheng, Fuyuan and Yu, Wangbo and Wang, Guangshuo and Luo, Guibo and Zhu, Yuesheng},
  booktitle={European Conference on Computer Vision},
  pages={477--493},
  year={2025},
  organization={Springer}
}
```

## Acknowledgement
Thanks to:
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [ModuleFormer](https://github.com/IBM/ModuleFormer/tree/main)
