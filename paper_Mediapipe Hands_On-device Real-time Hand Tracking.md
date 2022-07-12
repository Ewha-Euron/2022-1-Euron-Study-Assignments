# Mediapipe Hands: On-device Real-time Hand Tracking
    
- 논문: https://arxiv.org/pdf/2006.10214.pdf

## Overview

- 장치 내에 사람의 손 골격을 실시간으로 추적하는 솔루션
- 머신 러닝을 사용하여 단일 프레임에서 손의 21개 3D 랜드마크를 추론
- 파이프라인은 두 가지 모델로 구성됨
    
    1) palm detector
    
    2) hand landmark model
    

## Palm detection model

- 모바일 실시간 사용을 위한 single shot detector model
- 손바닥과 주먹과 같은 단단한 물체의 경계 상자를 추정하는 것이 손가락의 관절로 손을 감지하는 것보다 훨씬 간단함 → 손바닥 감지기를 학습시킴
- encoder-decoder feature extractor는 작은 개체에 대해서도 더 큰 장면의 context 인식을 위해 사용됨

## Hand landmark model

- 전체 이미지에 대한 손바닥 감지 후 hand landmark model은 회귀, 즉 좌표 예측을 통해 감지된 손 영역 내부의 21개 3D 손 관절 좌표의 정확한 키포인트 위치를 예측