# Learning Deep Features for Discriminative Localization

- CNN의 결과 설명
- Class Activation Maps(CAM): Convolution-Global average pooling -Softmax 구조
- Object loaclization도 함 

# MediaPipe Hands: On-device Real-time Hand Tracking

- 하드웨어 필요x, 2개 이상 손 탐지 및 일부 가려져도 탐지가능, 모바일 실시간 가능
- 1) Bounding Box를 찾는 palm detector
- 2) Bounding Box 당 21개 keypoints 감지
- 성능이 좋음

# Mask R-CNN 

- 이미지 내 instance segmentation mask 
- Mask R-CNN=Faster R-cNN + mask branch
- BackBone: ResNet & REsNeXt , Head: Faster R-CNN head 
- COCO 2016 challenge 1st
