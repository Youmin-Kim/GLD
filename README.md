# Distilling Global and Local Logits with Densely Connected Relations

Official Pytorch implementation of "Distilling Global and Local Logits with Densely Connected Relations", **ICCV 2021**.

| [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Distilling_Global_and_Local_Logits_With_Densely_Connected_Relations_ICCV_2021_paper.pdf) | [supplementary material](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Kim_Distilling_Global_and_ICCV_2021_supplemental.pdf) |

This repository contains source code of CIFAR-100 experimental setup (a). We provide a pre-trained teacher weight in "teacher" directory, median of 3 runs for starting distillation without pre-training the teacher network. Training logs of distilled student are in "log" directory.

Setup (a) : Teacher (ResNet-110), Student (ResNet-20).

## Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy

## Training a teacher network (If you need)
```
python3 ./train.py --model resnet --depth 110 
```

## Distilling the teacher network to the student network
```
python3 ./distill.py --teacher resnet --student resnet --depth 110 --sdepth 20 --alpha 0.7 --beta 500. --div 2
```

## Citation
```
@inproceedings{kim2021distilling,
  title={Distilling Global and Local Logits With Densely Connected Relations},
  author={Kim, Youmin and Park, Jinbae and Jang, YounHo and Ali, Muhammad and Oh, Tae-Hyun and Bae, Sung-Ho},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6290--6300},
  year={2021}
}
```
