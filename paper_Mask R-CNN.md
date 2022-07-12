# Mask R-CNN

- 논문: https://arxiv.org/pdf/1703.06870.pdf

## Overview

- 이미지 내에서 각 instance(object)에 대한 segmentation mask 생성 (classification + localization)
- Mask R-CNN은 Faster R-CNN에 mask branch를 추가한 구조
- 연구 목표는 instance segmentation task에서 사용 가능한 딥러닝 프레임워크를 개발하는 것

## Related Work

- R-CNN
- Instance segmentation

## Mask R-CNN
- Faster R-CNN의 RPN에서 얻은 ROI(Region of Interest)에 대해 pixel 단위의 segmentation mask를 예측하는 branch를 추가한 구조
- 객체의 class를 예측하는 classification branch, bbox regression을 수행하는 bbox regression branch와 독립적으로 segmentation mask를 예측하는 mask branch를 추가
- Mask R-CNN과 Faster R-CNN과 차이는 RoIAlign
    - Faster R-CNN은 pixel 단위의 segmentation을 위해 설계되지 않음
    - 이를 위해 RoIAlign이라는 간단하면서 정확한 공간 정보를 보존하는 레이어 제안
    - 이는 Faster R-CNN보다 mask의 정확도를 10~50% 높임

**Mask representation**

- mask는 input object의 spatial layout의 encode 결과
- class와 box의 위치 정보들이 FC layer를 거치면서 고정된 vector로 변환되기 때문에 공간적 정보 손실 발생
- mask는 convolution 연산에 의해 공간적 정보를 최소화시킬 수 있음
- fully convolution networks를 사용하여 각 RoI에 대해 m x m 사이즈의 mask를 예측
- mask는 FCN에 의해 1차원 벡터로 축소되지 않고, m x m 형태로 공간 정보 유지 가능
- FC layer보다 더 적을 파라미터 수가 사용되고, 연산 속도가 빠르고 정확함
- mask 정보를 m x m 형태로 보존하기 위해선, RoI feature가 요구됨 → 이를 위해 RoIAlign layer를 만듦