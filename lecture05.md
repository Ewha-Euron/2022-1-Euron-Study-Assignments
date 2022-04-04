## Lec5 - Convolutional Neural Networks

### History of CNN

- 1957년: 최초로 perceptron을 구현, 가중치 W를 업데이트하는 update rule
- 1960년: Multilayer Perceptron Network인 Adaline/Madaline
- 1986년: backpropagation, 신경망 학습 시작. But 다시 암흑기
- 2006년: Deep Learning, 초기화에는 RBM, hidden Layer에서 가중치가 학습.
- 2012년: ImageNet 분류-> AlexNet

### Fast-forward to today: CNNs
이미지 분류, 이미지 검색, detection과 segmentation, 자율주행, face recognition, pose recognition,...
- Image captioning: 사진을 보고 이미지에 대한 설명을 하는 것 (CNN+RNN)

### Convolutional Neural Networks

#### convolution Layer
convolve filter을 이용해 인풋의 이미지 특징을 뽑아냄(depth는 같아야함)
- w_T*x+b
=> 28*28*1 의 activation map => 하나의 층 생성 => ... 반복 => 이미지 특징 추출

#### conv-> ReLU->pooling-> ...
여기서 ReLU는 0 이하 값들을 전부 0으로 만들어 검정색이 됨 

#### activation map 과정
N: input size
F: filter 크기
output size = (N-F)/stride + 1
=> stride를 거칠수록 사이즈가 작아지는 문제 => `Padding`

#### 중간 summary
input size는 32, 64,128 ...
filter가 3이고 stride가 1이면 padding은 1
filter가 5이고 stride가 1이면 padding은 2

### Pooling
: 큰 특징값을 유지하면서 이미지 사이즈를 줄여주는 것 (주로 max pooling)


