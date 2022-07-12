# Learning Deep Features for Discriminative Localization

## Overview

- 위치에 대한 정보가 없는 GT(Classification에 대한 정보만 있는 Ground-Truth)이더라도 **GAP**(Global Average Pooling)을 활용하여 localization 능력을 줄 수 있음
    - conv layer에서 object를 localize 하는 능력이 뛰어남에도, 이 능력은 classification을 위한 FC layer에서 손실됨
    - 이때 마지막 layer까지 localization ability를 유지하기 위한 mechanism으로 GAP 제안
- **CAM**(Class Activation Mapping) 소개
- Bounding Box 정보에 대한 학습 없이 ILSVRC 2014에서 객체 위치 파악에 대해 37.1%(Top-5기준) 달성
- 해당 네트워크는 분류(Classification)에 대한 목표를 가지고 학습하더라도 이미지 영역을 localization 할 수 있음

## Related work

- CAM 등장 전에는 end-to-end 구조가 존재하지 않았음
- CAM을 사용할 경우 추가적인 모델 수정 없이 weight와 feature map을 활용하여 localization을 할 수 있음 → end-to-end
- CAM 대신 GMP(global max pooling)나 GAP 방법을 활용하여 localization을 시도
    
    → 이 방법은 한계가 있음. 물체의 전체를 보는 게 아니라 경계부분만 봄
    

## Class Activation Mapping (CAM)
- 모델의 전체적인 구조: conv layer → GAP → softmax
- "a weighted sum of the presence of visual patterns at difference spatial location”
- CAM은 마지막 convolutional feature map의 내적(dot product)과 FC-layer의 Class별 Weight의 합

## Global Average Pooling(GAP) vs. Global Max Pooling(GMP)

- GMP와 GAP 모두 classification에서 정확도는 비슷함
- 하지만 localization에서는 GAP >> GMP
    - GMP는 feature map에서 하나의 뚜렷한 특징을 찾아내고, GAP는 이미지 전체적으로 뚜렷한 특징이 있는지를 찾아냄

## Results

- 비슷한 카테고리의 이미지들에서는 비슷한 객체가 주로 추출됨
- 추상적인 설명이 label로 제공된 이미지에 대해서도 해당 정보가 포함된 위치를 잘 포착해냄