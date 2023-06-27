PyTorch Code for the following paper at CVPR2023:
Title:OT-Filter: An Optimal Transport Filter for Learning with Noisy Labels
Authors:Chuanwen Feng, Yilong Ren, Xike Xie
Institute: University of Science and Technology of China(USTC)
https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_OT-Filter_An_Optimal_Transport_Filter_for_Learning_With_Noisy_Labels_CVPR_2023_paper.pdf

Requirements:
python 3.9
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install scikit-learn
pip install pot

Experiments:
python Train_CIFAR.py

Cite OT-Filter
If you find the code useful in your research, please consider citing our paper:
@InProceedings{Feng_2023_CVPR,
    author    = {Feng, Chuanwen and Ren, Yilong and Xie, Xike},
    title     = {OT-Filter: An Optimal Transport Filter for Learning With Noisy Labels},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {16164-16174}