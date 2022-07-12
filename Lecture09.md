# CS231N Lecture9 | CNN Architectures

> ImageNet 챌린지에서 우승한 모델들로, 대표적인 CNN 모델들(AlexNet, VGGNet, GoogleNet, ResNet)을 살펴볼 것이다. 이외에도 잘 사용하진 않지만 역사적으로 중요한 모델이나, 흥미로운 모델, 최신(2017 기준) 모델들도 다룰 것이다.

> 앞으로 말하는 ‘네트워크가 깊다’는 것은 “학습 가능한 가중치를 가진 레이어”의 개수가 많은 것을 의미한다. 가령 conv layer나 FC layer를 말하고, 이때 pooling layer는 포함되지 않는다.

## 0. LeNet

- 실제로 성공적으로 사용된 최초의 conv net
- 숫자 인식에서 엄청난 성공을 거둠

## 1. AlexNet (2012)
![Untitled](https://user-images.githubusercontent.com/79077316/167327738-7f942639-e68c-423f-8494-8140511142cb.png)

- 최초의 large scale CNN으로 LeNet-5와 기본 구조가 크게 다르지 않음
    - 2개의 GPU로 병렬연산을 수행하기 위해 병렬적인 구조로 설계된 것이 가장 큰 변화
- ImageNet의 classification task에서 우승한 모델
- CNN의 부흥을 일으킨 모델
- 다양한 task의 transfer learning에 많이 사용되어 옴
- 5개의 conv layer와 3개의 FC layer로 구성됨
    - conv1 - max pool1 - norm1 - conv2 - max pool2 - norm2 - conv3 - conv4 - conv5 - max pool3 - fc6 - fc7 - fc8
    - input 이미지의 크기는 227x227x3
    - 첫번째 layer의 출력 사이즈: 55x55x96(width x height x 필터의 개수)
    - parameter의 수: (11x11x3)x96 = 35K개
    - 두번째 layer의 출력 사이즈: 27x27x96(depth의 크기는 변하지 않음)
- conv1 (첫 번째 layer)
- pooling layer
    - 파라미터가 없음
    - 가중치가 없고 특정 지역에서 가장 큰 값을 추출하는 역할만 함
- FC layer
    - 4096개의 노드를 가진 레이어
    - FC8은 softmax 함수 사용

### Details/restropectives

- 활성화함수로 ReLU를 사용
- Data augmentation을 엄청 함 (flipping, jittering, color norm 등등)
- Drop out 0.5 - 학습 시 batch size는 128
- SGD momentum 0.9
- 초기 learning rate은 1e-2이고, val accuracy가 올라가지 않는 지점에서는 learning rate을 1e-10까지 줄임
- L2 weight decay
- 모델 앙상블로 성능을 향상시킴
- AlexNet 논문의 아키텍처와 관련된 이슈 - 위 그림에서는 첫 레이어가 224x224라고 되어 있지만 실제 입력은 227x227

다른 Conv Net과의 차이점 -> 모델이 두 개로 나눠져서 서로 교차함

- 모델의 기본 구조가 병렬 구조로 이루어져 있으며 두 개로 나눠져서 서로 교차함 → 네트워크를 GPU에 분산시켜서 넣음
    - 첫 번째 레이어의 출력이 55x55x96이고, 각각의 GPU에서의 depth가 48
    - 두 개의 GPU에서 feature map을 절반씩 나눠가짐
    - conv1, 2, 4,5에서는 같은 GPU 내에 있는 feature map(48개)만 사용
    - 반면에 conv3, FC6, 7, 8은 전체 feature map(98개)과 연결되어 있음

## 2. VGGNet (2014)
![2](https://user-images.githubusercontent.com/79077316/167327740-cf7e068a-1e85-4aab-8f32-d1534e41b379.png)
- 2014년도 ImageNet challenge의 classification task에서 2등한 oxford의 모델
- localization challenge 등 다른 task에서는 1등한 모델
- 이전 모델들과 비교했을 때 네트워크가 훨씬 기어지고 더 작은 필터를 사용함
- AlexNet에서는 8개의 레이어였지만 VGGNet에서는 16-19개의 레이어를 가짐
- 3x3 사이즈의 아주 작은 크기의 필터만 사용
    - 필터의 크기가 작으면 파라미터의 수가 더 적어지고, 큰 필터를 사용했을 때보다 레이어를 더 많이 쌓아서 네트워크를 더 깊게 만들 수 있음(depth⬆)
- 3x3 필터를 3개 쌓은 것은 결국 7x7 필터를 사용하는 것과 동일한 receptive field를 가짐
    - receptive field는 filter가 한 번에 볼 수 있는 입력의 sparical area
    - 7x7 필터와 동일한 receptive field를 가지면서도 더 깊은 레이어를 쌓게 됨
    - 네트워크를 더 깊게 함으로써 non-linearity를 더 추가할 수 있고, 파라미터의 수도 더 적어짐 (depth c인 네트워크에 대해 3x3 필터를 3개 사용할 경우 파라미터의 수는 3x3xCxCx3, 7x7 필터인 경우 7x7xCxC)
- 네트워크가 깊다는 것은 학습 가능한 가중치를 가진 레이어의 개수가 많다는 것을 의미

### **Details**

- Local Response Normalization 사용 안 함
- 성능 향상을 위해 앙상블 기법 사용
- VGG19가 메모리를 조금 더 쓰지만 성능이 좀 더 좋음
- 보통 VGG16을 더 많이 사용함

## 3. GoogleNet (2014)
![](https://user-images.githubusercontent.com/79077316/167327742-8d48da73-b2c7-477c-83c5-fb7e4ab40eeb.png)

- 2014년도 ImageNet challenge 우승 모델
- 22개의 layers
- **Inception module**을 여러 개 쌓아서 만듦
- 파라미터의 수를 줄이기 위해 FC-layer를 없앰
- 네트워크가 훨씬 깊은데도 전체 파라미터 수는 5M 정도로 AlexNet(60M)보다 적음

![](https://user-images.githubusercontent.com/79077316/167327745-5753410f-2302-4b77-979c-4eb66588f0fe.png)

- “network within a network”라는 개념으로 local topology를 구현
- 각각의 local network가 Inception module
- 위 그림은 기본적인 Inception module(naive version)
- 내부에는 동일한 입력을 받는 서로 다른 다양한 크기의 필터들이 **병렬적**으로 존재함
    - 이는 다양한 feature를 뽑기 위해 여러 종류의 convolution filter를 병렬적으로 사용한 것임
- 다양한 연산을 수행하고 이를 하나로 합치는 단순한(naive) 방식에는 여러 문제가 있음
    - 계산 비용의 문제. 연산량이 매우 많아짐
    - pooling layer에서 depth가 그대로 유지돼서 레이어를 거칠 때마다 다른 레이어의 출력이 계속해서 더해져서 depth가 점점 커짐  
    👉🏻 **bottleneck layer**를 사용해서 해결!
    

**bottleneck layer**

- conv 연산을 수행하기 전에 입력을 더 낮은 차원으로 보냄
- 입력의 depth를 더 낮은 차원으로 projection 함
- Feature depth를 줄이기 위해 1x1 conv layer 사용
- input feature map들 간의 선형결합(linear combination)이라고 할 수 있음

![](https://user-images.githubusercontent.com/79077316/167327751-fbc93d44-361e-4a0e-88a2-67025d38ee2a.png)

- GoogleNet의 앞단(stem)은 일반적인 네트워크 구조
- 가중치를 가진 학습 가능한 레이어의 개수는 22개
- 각 Inception module은 1x1/3x3/5x5 conv layer를 병렬적으로 가지고 있음
- 앞단 이후에 Inception module을 쌓아올리고 마지막에 classifier 결과를 출력
- 보조분류기(auxiliary classifier)가 추가됨
    - average pooling과 1x1 conv가 있고 FC-layer도 있음
    - softmax로 ImageNet 클래스 분류
    - ImageNet trainset loss 계산
    - 네트워크가 깊어서 네트워크의 끝 뿐만 아니라 보조분류기에서도 loss를 계산함
    - 보조분류기를 추가함으로써 그레디언트를 얻게 되고, 중간 레이어의 학습을 도움

## 4. ResNet (2015)

![](https://user-images.githubusercontent.com/79077316/167327757-24af8186-bdfd-4b7b-86bf-dde6b63632a5.png)

- 2015 ILSVRC 우승한 모델
- 152개의 layer로 네트워크의 깊이가 매우 깊은 모델
- DNN에서 layer를 깊게 쌓을수록 성능이 더 좋아질 것이라고 예상했지만, 실제로는 20개 이상부터는 성능이 낮아지는 degradation 문제 발생
- ResNet은 Residual learning이라는 방법을 통해 모델의 층이 깊어져도 학습이 잘 되도록 구현한 모델

### Residual Learning
- 56 Layer의 CNN과 20 Layer의 CNN을 비교하여 테스트했을 때, test error가 56 Layer의 CNN이 더 높았음
- training error 역시 56 Layer CNN이 더 안 좋은 것으로 보아 성능이 낮은 이유가 overfitting 때문이 아님을 알 수 있음
- ResNet 저자들이 내린 가설은 더 깊은 모델 학습 시 optimization에 문제가 생긴다는 것. 즉 모델이 깊어질수록 최적화가 어렵다.
- “네트워크의 깊이가 더 깊다면 적어도 더 얕은 모델만큼의 성능은 나와야 한다” 이 추론을 이용해서 더 얕은 모델의 가중치를 더 깊은 모델의 일부 레이어에 복사. 나머지 레이어는 identity mapping(input을 그대로 output으로). 이렇게 하면 적어도 shallower model만큼의 성능은 보장됨
- 레이어를 단순하게 쌓지 않음으로써 위 motivation을 모델 아키텍처에 적용

![](https://user-images.githubusercontent.com/79077316/167327759-3a86d173-b7e1-46dc-9c48-a357a636a174.png)
👉🏻 Direct mapping 대신 Residual mapping을 하도록 블럭을 쌓는다!

- H(x)를 바로 학습하기보다 H(x)-x를 학습하도록! -> 이를 위해 skip connection 도입!
- 오른쪽 그림의 skip connection은 가중치가 없고, 입력은 identity mapping으로 그대로 output으로 내보냄
- 최종 출력값인 H(x)=F(x)+x에서 x는 input이 되고, F(x)는 변화량(delta, residual). 즉 최종 출력 값은 **input X + 변화량(Residual)**
- 결국 네트워크는 변화량(residual)만 학습하면 됨. H(x)를 직접 학습하는 대신 Residual를 학습하는 것이 훨씬 쉬울 것이라고 생각.
- Identity Mapping에 가까운 값을 얻기 쉬움 ( F(x)=0 )
- ResNet은 이처럼 residual block들을 쌓아 올린 구조

### Details

![](https://user-images.githubusercontent.com/79077316/167327764-48896a15-6165-4670-abec-f620cfc74772.png)
- 각각의 Residual block은 두 개의 3x3 conv layers로 이루어져 있음
- 주기적으로 필터를 두 배씩 늘리고 stride 2를 이용해서 downsampling을 수행
- 네트워크 초반에는 Conv-layer가 추가적으로 있고 네트워크 끝에는 FC-layer가 없음
- Global Average Pooling을 사용
    - 하나의 Map 전체는 Average pooling
- 마지막에는 1000개의 클래스 분류를 위한 노드가 붙음
- ResNet의 경우 모델의 depth가 50 이상일 때 GoogleNet과 유사하게 Bottleneck Layer를 도입함
    - 1x1 Conv layer를 도입하여 초기 필터의 depth를 줄여줌
- ResNet은 Conv layer 뒤에 모두 Batch Norm을 사용
- 가중치 초기화는 Xavier/2를 사용
- SGD + Momentum (0.9)
- Minibatch size 256
- weight decay 적용
- Drop out은 하지 않음

## 5. 모델별 Complexity

![](https://user-images.githubusercontent.com/79077316/167327773-da920b24-fbb6-47eb-90af-c6d3e623bd7c.png)

- 오른쪽 그래프에서 원의 크기는 메모리 사용량을 나타냄
- VGG - 가장 효율성이 떨어짐. 메모리 사용량도 크고 연산량이 많음
- AlexNet - 계산량이 작지만, 메모리 사용량이 크고 비효율적. 정확도도 낮은 편
- ResNet - 메모리 사용량과 계산량은 중간 정도이지만 accuracy는 최상위