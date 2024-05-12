# ACCoNet
This project provides the code and results for 'Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images', IEEE TCYB, 2023. [IEEE link](https://ieeexplore.ieee.org/document/9756652) and [arxiv link](https://arxiv.org/abs/2203.13664) [Homepage](https://mathlee.github.io/)

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/ACCoNet/blob/main/image/ACCoNet.png">
   </div>
   
   
# Requirements
   python 2.7 + pytorch 0.4.0 or
   
   python 3.7 + pytorch 1.9.0


# Saliency maps
   We provide saliency maps of our ACCoNet ([VGG_backbone](https://pan.baidu.com/s/11KzUltnKIwbYFbEXtud2gQ) (code: gr06) and [ResNet_backbone](https://pan.baidu.com/s/1_ksAXbRrMWupToCxcSDa8g) (code: 1hpn)) on ORSSD, EORSSD, and additional [ORSI-4199](https://github.com/wchao1213/ORSI-SOD) datasets.
      
   ![Image](https://github.com/MathLee/ACCoNet/blob/main/image/table.png)
   
# Training

We provide the code for ACCoNet_VGG and ACCoNet_ResNet, please modify '--is_ResNet' and the paths of datasets in train_ACCoNet.py.

For ACCoNet_VGG, please modify paths of [VGG backbone](https://pan.baidu.com/s/1YQxKZ-y2C4EsqrgKNI7qrw) (code: ego5) in /model/vgg.py.

data_aug.m is used for data augmentation.


# Pre-trained model and testing
1. Download the following pre-trained models and put them in /models.

2. Modify paths of pre-trained models and datasets.

3. Run test_ACCoNet.py.

ORSSD: [ACCoNet_VGG](https://pan.baidu.com/s/1mPb7oyaz9OVKs3T9v4xCmw) (code: 1bsg); [ACCoNet_ResNet](https://pan.baidu.com/s/1UhHLxgBvMgD66jz2SKgclw) (code: mv91).

EORSSD: [ACCoNet_VGG](https://pan.baidu.com/s/1R2mFox8rEyxH1DTTnMinLA) (code: i016); [ACCoNet_ResNet](https://pan.baidu.com/s/1-TkZcxR6fBNYWKljhL1Qrg) (code: ak5m).

ORSI-4199: [ACCoNet_VGG](https://pan.baidu.com/s/1WUVmVCwICBEM3gUJxQ5pkw) (code: qv05); [ACCoNet_ResNet](https://pan.baidu.com/s/1I4RWaLDx4ukK8_11y1AEtw) (code: art7).

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2023_ACCoNet,
                author = {Gongyang Li and Zhi Liu and Dan Zeng and Weisi Lin and Haibin Ling},
                title = {Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images},
                journal = {IEEE Transactions on Cybernetics},
                volume = {53},
                number = {1},
                pages = {526-538},
                year = {2023},
                month = {Jan.},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
