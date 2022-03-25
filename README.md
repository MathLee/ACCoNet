# ACCoNet
[TCYB2022] [ACCoNet] Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images

# CorrNet
This project provides the code and results for 'Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images', IEEE TCYB, accepted, 2022.

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/ACCoNet/blob/main/image/ACCoNet.png">
   </div>
   
   
# Requirements
   python 2.7 + pytorch 0.4.0 or
   
   python 3.7 + pytorch 1.9.0


# Saliency maps
   We provide saliency maps of [our ACCoNet] on ORSSD and EORSSD datasets.
   
   In addition, we also provide [saliency maps of our ACCoNet] on the recently published [ORSI-4199](https://github.com/wchao1213/ORSI-SOD) dataset.
   
   ![Image](https://github.com/MathLee/ACCoNet/blob/main/image/table.png)
   
# Training

Modify pathes of [VGG backbone](https://pan.baidu.com/s/1YQxKZ-y2C4EsqrgKNI7qrw) (code: ego5) in /model/vgg.py and datasets, then run train_ACCoNet.py.


# Pre-trained model and testing
Download the following pre-trained model, and modify pathes of pre-trained model and datasets, then run test_ACCoNet.py.

We also uploaded these pre-trained models in /models.

[ORSSD]

[EORSSD]

[ORSI-4199]

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2022_ACCoNet,
                author = {Gongyang Li and Zhi Liu and Dan Zeng and Weisi Lin and Haibin Ling},
                title = {Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images},
                journal = {IEEE Transactions on Cybernetics},
                year = {2022},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
