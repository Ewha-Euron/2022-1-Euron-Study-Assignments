# CS231n Lecture 5 | Convolutional Neural Networks

## 1. Neural Networks의 역사

### 1) Hubel & Wiesel의 연구

- 자세한 연구 내용은 1강을 참고

![Untitled (5)](https://user-images.githubusercontent.com/79077316/161549788-a63a19bd-c1b1-4899-bc2d-310300a72c5c.png)

**실험에서 알게 된 것**

1. 실험에서 뉴런이 oriented edges와 shapes와 같은 것에 반응한다는 것을 알게 됨
2. 실험 결과 대뇌 피질 내부에 지형적인 매핑(topographical mapping)이 존재하는 것을 알게 됨
3. 피질 내 서로 인접해 있는 세포들은 visual field 내에 지역성을 띄고 있음
4. 뉴런들이 계층적인 구조를 지님
    - simple cell, complex cell, hypercomplex cell 모두 계층적으로 연결되어 있음
    - simple cell
        - 다양한 edges의 방향과 빛의 방향에 반응
    - complex cell
        - 빛의 방향 뿐만 아니라 움직임에서 반응
    - hypercomplex cell
        - 끝 점(end point) 과 같은것에 반응
    
    👉🏻 물체의 corner 또는 blob에 대한 아이디어를 얻게 됨
    

### 2) Neocognitron (Fukushima, 1980)

![Untitled (6)](https://user-images.githubusercontent.com/79077316/161549800-b32e5ceb-4d82-4d59-9d3c-55bc69b15340.png)

- hubel과 wiesel의 연구에서 발견한 simple cell과 complex cell의 아이디어를 사용한 최초의 NN
- simple cell과 complex cell을 교차시킴 (SCSC...)
- Simple cells은 학습가능한 parameters를 가지고 있고, Complex cells은 pooling과 같은 것으로 구현하여 작은 변화에 Simple cells보다 더 강인함

### 3) LeNet, 1998

![Untitled (7)](https://user-images.githubusercontent.com/79077316/161549814-41d61b7a-4ed1-4a16-bce9-a4b3c97c3f14.png)

- 최초로 NN을 학습시키기 위해 Backprop과 gradient-based learning을 적용함
- 문서 인식에서 꽤 좋은 성능을 보임 (우편번호의 숫자 인식)

### 4) AlexNet, 2012

![Untitled (8)](https://user-images.githubusercontent.com/79077316/161549880-5342c913-18eb-4575-aa80-ad1cb5a6e0d9.png)

- convnet으로 이전의 NN보다 더 크고 깊어짐
- GPU의 발전으로 ImageNet dataset과 같은 대규모의 데이터도 활용할 수 있게 됨

## 2. Convolutional Neural Networks

### 1) Conv layer의 연산 과정

![Untitled (9)](https://user-images.githubusercontent.com/79077316/161549867-bb737fd0-e3df-46bf-9290-3af19f251d5c.png)

- FC layer와 Conv layer의 차이점은 기존의 구조를 보존시킨다는 것임
    - FC layer는 입력 이미지를 벡터 한 줄로 길게 폈다면 conv layer는 기존의 이미지를 그대로 유지함
- 가중치 값이 들어있는 filter를 이용해서 이미지를 슬라이딩하면서 내적 연산을 수행함
- filter는 입력의 깊이(depth)만큼 확장됨
- 5x5x3의 필터는 벡터를 길게 펴서 1x75 길이의 벡터가 됨
- 필터의 각 w와 이에 해당하는 이미지의 픽셀 값을 곱함
    - W^Tx + b
    - W를 transpose한 값과 입력 이미지의 픽셀 값을 내적 연산하고 bias 값을 더함

### 2) Convolution 연산

![Untitled (10)](https://user-images.githubusercontent.com/79077316/161549892-a45bc997-3595-45b1-a94a-a29ff00af0a2.png)

- convolution 연산은 입력 이미지의 좌상단부터 시작하여 필터의 모든 요소를 가지고 내적을 수행하여 하나의 값을 얻게 됨
- 하나의 필터를 가지고 이미지 전체에 대해 convolution 연산을 수행해서 출력 결과 activation map을 얻음

![Untitled (11)](https://user-images.githubusercontent.com/79077316/161549914-0b98d25a-a52a-4ee8-b26e-dbb3105099cd.png)

- 보통 위 그림과 같이 convolution layer에서는 필터마다 다른 특징을 추출해내기 위해 여러 개의 필터를 사용함

![Untitled (12)](https://user-images.githubusercontent.com/79077316/161549825-887ddbb6-5821-4c51-b021-56c34f79e22a.png)

- conv layer들을 연산하고 각각을 쌓아 올리게 되면 위와 같이 간단한 linear layer들이 여러 겹 쌓인 neural network가 됨

![Untitled (13)](https://user-images.githubusercontent.com/79077316/161549924-be90d83c-7aab-421a-9f5b-5557f297df6d.png)

- conv layer 사이사이에 activation function이나 pooling layer을 추가함
- 각 layer의 출력은 그 다음 layer의 입력이 됨
- 각 layer는 여러 개의 필터를 가지고 있고, 각 필터마다 각각의 activation map을 만듦

![Untitled (14)](https://user-images.githubusercontent.com/79077316/161549933-13b9f67d-578c-4fc0-9333-994eec8f555e.png)

- hubel & wiesel의 연구 결과처럼 여러 개의 layer들은 각 필터를 거치면서 계층적으로 학습함
- 앞쪽에 있는 필터에서는 edge와 같은 low-level feature를 학습함
- mid-level에서는 corner나 blobs와 같은 좀 더 복잡한 특징을 학습함
- high-level에서는 좀 더 객체와 닮은 것들이 출력 결과로 나옴
- layer의 계층에 따라 단순/복잡한 특징이 존재함

![Untitled (15)](https://user-images.githubusercontent.com/79077316/161549944-09ff394d-8003-43dd-abc3-1e16569e6704.png)

- 각 activation은 이미지가 필터를 통과한 결과이고, 이미지 중 어느 위치에서 해당 필터가 크게 반응하는지 알려줌

### 3) Activation map의 연산 과정

![Untitled (16)](https://user-images.githubusercontent.com/79077316/161549955-4c02d48b-5ad8-4d3e-9beb-fcdcf20bdeb8.png)

- 간단한 예시로 7 x 7 입력에 3 x 3 필터가 있다고 해보자
- 3x3 필터를 이미지의 왼쪽 상단부터 씌워서 해당 값들의 내적을 수행함
- 위 이미지의 연산 결과는 activation map의 가장 왼쪽, 윗부분의 출력값이 됨
- 연산이 끝나면, 필터를 오른쪽으로 한칸씩 움직여서 반복 연산을 수행함
- 이렇게 반복하면 결국 5x5의 출력을 얻게 됨

![Untitled (17)](https://user-images.githubusercontent.com/79077316/161549971-477cf5cc-c88c-4e02-bbed-a7366c8a9650.png)

- 필터를 이동하는 칸 수를 stride라고 함
- stride=2인 경우 출력은 3x3이 됨
- stride=3인 경우 이미지의 사이즈에 맞아 떨어지지 않아 제대로 연산이 동작하지 않음

![Untitled (18)](https://user-images.githubusercontent.com/79077316/161549980-a9b3c778-6325-48d4-aeea-6c25b2d5cffd.png)

- 스트라이드의 크기에 따라 출력 사이즈가 어떻게 될 것인지 알려주는 수식
- 위 수식을 이용해서 어떤 크기의 필터를 사용해야 하는지, stride를 몇으로 했을 때 이미지에 꼭 맞는지, 출력의 사이즈는 어떻게 되는지 알 수 있음
- stride가 클수록 출력은 점점 작아짐

**zero-padding**

- 출력의 사이즈를 조절하는 방법 → 입력 사이즈와 출력 사이즈를 같도록 하기 위함
- 바깥쪽 부분의 이미지 픽셀값이 상대적으로 덜 강조되는 문제를 해결하기 위한 방법
- 이미지의 가장 자리에 0을 채워 넣는 방법

![Untitled (19)](https://user-images.githubusercontent.com/79077316/161549991-3ff7db5d-b494-442a-82b5-4472e4ed2246.png)

- zero-padding을 적용하면0 N=9를 넣어 수식을 사용하면 됨
- 위 예제의 경우 출력의 사이즈는 7x7x(필터의 개수)

### 4) Brain Neuron 관점에서의 Conv layer

![Untitled (20)](https://user-images.githubusercontent.com/79077316/161549998-c2f8c581-a54e-4eed-9662-eb8d73fa4dfa.png)

- 전체 이미지의 특정 위치에 필터를 놓고 내적을 수행
- 뇌의 뉴런과 다른 점은 우리 뇌의 뉴런은 local connectivity를 가지고 있음
    - conv layer처럼 슬라이딩을 해서 모든 부분과 연산을 하는 게 아니라, 우리 뇌의 뉴런은 특정 부분에만 연결되어 있음
    - 하나의 뉴런은 한 부분만 처리하고, 뉴런들이 여러 개가 모여서 전체 이미지를 처리하는 것임
- 위 이미지에서 한 뉴런의 receptive field는 5x5임
    - receptive field란 한 뉴런이 한 번에 수용할 수 있는 영역을 의미

![Untitled (21)](https://user-images.githubusercontent.com/79077316/161550009-9e8b6094-47dd-4553-839c-c7be4aec2775.png)

- 출력의 사이즈는 28x28x(필터의 개수)
- 파란색 volume 안에 5개의 점은 정확하게 같은 지역에서 추출된 서로 다른 특징임

### 5) Pooling layer

![Untitled (22)](https://user-images.githubusercontent.com/79077316/161550021-223ac637-845e-4cb5-a0a8-bc6cc4e059ef.png)

- CNN에는 conv layer 말고도 pooling layer가 있음
- pooling layer는 representation을 더 작게 만듦
    - representation을 작게 하면 파라미터의 수가 줄게 됨
    - 공간적인 불변성도 얻을 수 있음
- 결국 pooling layer가 하는 일은 downsampling
- 단, depth는 변하지 않음

![Untitled (23)](https://user-images.githubusercontent.com/79077316/161550027-b42a6ea8-571c-4f9e-83f9-d96b4af5ced0.png)

- 일반적으로 max pooling이 많이 쓰임
- 2x2 필터이고, stride=2일 때 필터 안에 가장 큰 값을 고르면 됨
- conv layer 연산과 달리 pooling 할 때는 겹치지 않도록 하는 것이 일반적임