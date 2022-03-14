# CS231n 2강 Image classification (knn과 linear classification)

## 1. Image Classification

- 인풋 이미지가 주어진 클래스 중 어떤 클래스에 속하는지 분류하는 태스크

### Semantic gap

- 기계 입장에서 image classification이 어려운 이유는 사람이 보는 이미지와 기계가 보는 이미지가 다르기 때문
- 사람 ↔ 기계가 이미지를 받아들이는 방식에 의미론적인 차이가 존재 (semantic gap)
- 컴퓨터는 이미지를 **숫자 집합**으로 인식
    - 이미지의 각 픽셀이 r, g, b 3개의 숫자로 표현됨

### Challenges

- 이미지 분류를 어렵게 하는 여러 challenge가 존재함
    - viewpoint variation(바라보는 시점), illumination(조명), deformation(이미지 속 객체의 변형), occlusion(가려지는 현상), background clutter(배경과 비슷해 보이는 현상), intraclass variation(클래스 내의 다양성)
- 예를 들어, 고양이 사진의 viewpoint가 달라지면 이미지의 픽셀 값도 모두 달라지고 기계가 보는 이미지는 완전히 다른 이미지가 되는 것임. 그럼에도 이미지를 여전히 ‘고양이’로 분류할 줄 알아야 함
- 사람의 시각체계와 달리 위의 모든 challenge들을 극복하고 컴퓨터가 이미지를 잘 분류해내는 건 기적에 가까움
- 일부 제한된 상황을 가정하면 이미지 분류가 가능해짐
    
    👉🏻 어떻게 가능해질까?
    

### Edge를 이용해 이미지를 분류하는 방법

- Hubel과 Wiesel의 연구에서 Edge가 이미지를 인식할 때 중요한 역할을 한다는 것을 알게 됨
- 이미지에서 edges를 계산하고, 다양한 corner와 edge를 분류해서 이미지를 분류함
    - 예를 들어 고양이 이미지에서 edge를 추출해내고, ‘Y’ 모양이면 고양이의 코로 분류함
- 하지만 이런 방식은 문제가 있는데,
    1. 알고리즘이 robust하지 않음 (즉 이미지에 변화가 있을 때 다른 클래스로 잘못 분류함)
    2. 확장성이 없음 (고양이를 분류하는 규칙을 직접 만들어 분류하는 것이기 때문에 고양이가 아닌 다른 객체는 인식하지 못함)
- 위와 같은 문제로 data-driven approach(데이터 중심 접근 방법)으로 옮겨 감

### Data-driven approach 데이터 중심 접근 방법

- 위에서 직접 규칙을 만들어 나가는 대신에
    1. 엄청 많은 데이터셋을 수집하고, 
    2. 데이터셋을 이용해서 machine learning classifier를 학습시켜 이미지 분류 모델을 만들어 냄
    
    👉🏻 따라서 함수 두 개가 필요함 train, test
    
- 이미지를 분류할 때 사용되는 굉장히 general한 방법임
    1. Collect a data set of images and labels
    2. Use Machine Learning to train a classifier
    3. Evaluate the classifier on new images

```python
def train(images, labels):
	#machine learning!
	return model

def predict(model, test_images):
	#use model to predict labels
	return test_labels
```

- 데이터 중심 접근 방법은 머신 러닝의 key insight!

## 2. Nearest Neighbor

- 굉장히 간단한 알고리즘이지만 data-driven approach로서 아주 좋은 방법임
    1. 모든 학습 데이터를 기억하고
    2. 새로운 이미지(test image)와 기존의 학습 데이터를 비교해서 **가장 유사한** 이미지로 레이블을 예측함
    
    👉🏻 이때 가장 유사한 것의 기준을 무엇이며, 두 이미지를 어떻게 비교할까? 다른 말로 **“이미지 간의 차이를 어떻게 측정할 것인가?”**
    
    ### L1 Distance
    
    ![Untitled](https://user-images.githubusercontent.com/79077316/158099511-e3415d96-1a29-4ace-a107-a674de5e7141.PNG)
    
    - L1 distance는 같은 point의 input 이미지(test set)의 픽셀에서 training image의 픽셀 값을 빼고 절댓값을 취함
    - 이렇게 픽셀 간의 차이를 계산하고 모든 결과를 더한 것이 L1 distance의 결과값이 됨
- `numpy`의 벡터 연산을 이용하면 NN Classifier를 구현하는 코드는 간결해짐

![Untitled](https://user-images.githubusercontent.com/79077316/158099513-2e4ef171-12e7-415c-9aa8-a1a588f0a56d.PNG)

- training set의 이미지가 N개일 때, train 함수의 시간 복잡도는 O(1)이고, test 함수의 시간 복잡도는 O(N)
- test time에서 N개의 학습 데이터를 test image와 모두 비교하면서 test time >> train time
- 실제로 우리는 train time 보다는 test time에서 빠르게 동작하기를 원함
- 하지만 NN 알고리즘은 정반대로 test time에서 시간이 많이 걸림
- CNN과 같은 parametric model은 NN과 반대로 train time에서는 시간이 오래 걸릴지 모르지만, test time은 엄청 빠름 (뒤에서 소개할 linear classification도 parametric 방법)

### NN의 decision regions

![Untitled](https://user-images.githubusercontent.com/79077316/158099517-4fb5e2c7-36f8-4610-8be7-f3270b2a8591.PNG)

- 🙄 nn의 decision regions가 어떻게 그려지는 건지 아직 제대로 이해를 못했다. 좀 더 공부하고 추가할 예정!
- 위 그림의 결과를 보면 성능이 그닥 좋지 않음
    1. 가운데에 노란 점. nn은 ‘가장 가까운 이웃’만을 보기 때문에 생기는 문제(🙄 이게 왜..?)
    2. noise나 spurious
- 위의 문제들 때문에 nn의 조금 더 일반화된 방법인 knn 알고리즘이 탄생

## 3. K-Nearest Neighbor

![Untitled](https://user-images.githubusercontent.com/79077316/158099521-e6b018a9-9374-4100-9f00-c271ed5ceb34.PNG)

- 단순하게 가장 가까운 이웃만을 찾기보다는, distance metric을 이용해서 가까운 이웃을 k개 찾고, 이웃끼리 다수결 투표를 함. 이웃 들 중 가장 많은 득표수를 획득한 클래스의 레이블로 예측하는 방법
- 이때 k는 1보다 커야 함
- 위 그림은 모두 동일한 데이터셋을 사용한 knn 분류기
- nn(k=1)보다 경계가 부드러워지고, nn에서 문제가 되는 점들도 보이지 않음. 그림에서의 흰색 영역은 knn이 ‘majority’를 결정할 수 없는 영역

### Distance metric

- knn에서 서로 다른 점(이미지)들을 어떻게 비교할 것인가?
    
    👉🏻 “Distance metric” 거리 척도를 사용해서 거리로 두 점을 비교한다!
    

![Untitled](https://user-images.githubusercontent.com/79077316/158099529-12822681-3a18-44a6-bde8-abb25b53dfdd.PNG)

- 🙄 L1 Distance와 L2 Distance 관련 내용이 잘 이해가 가지 않아서 이것 역시 더 공부하고 추가할 예정이다. 아래 내용들이 잘 이해되지 않음...
- 기존의 좌표계를 회전시키면 L1 distance는 변하는 반면 L2 distance는 좌표계와 아무런 연관이 없음
- 동일한 데이터에 대해 어떤 거리 척도를 사용하느냐에 따라 결정 경계(decision boundaries)의 모양이 달라짐
- L1 distance ⇒ 결정 경계가 coordinate axis(좌표 축)에 영향을 받음
- L2 distance ⇒ 좌표 축에 영향을 받지 않고 결정 경계를 만들기 때문에 더 자연스러움

### Hyperparameter

- knn을 사용할 때 결정해야 하는 것들이 있음
    1. k를 몇으로 설정할 것인지
    2. 어떤 거리 척도를 사용할 것인지
    
    **👉🏻** 이런 것들을 **“hyperparameter”** 라고 함
    
- 하이퍼파라미터는 train time에 학습하는 것이 아니므로 train time 전에 반드시 설정해야 함
- 하이퍼파라미터는 데이터로 직접 학습시킬 수 있는 것이 아님
    
    **👉🏻 그래서 하이퍼파라미터를 어떻게 정하냐?**
    
- 하이퍼파라미터를 정하는 것은 문제의존적(problem dependent)이다. 즉 문제마다 다르다
- 가장 간단한 방법으로는 데이터에 맞게 여러 값들을 시도해보고 가장 좋은 값을 찾는 것임
    
    **👉🏻 이건 구체적으로 어떻게 시도할 수 있을까?**
    

1️⃣ **하이퍼 파라미터를 구하는 첫 번째 방법**

- 데이터를 training set, validation set, test set 세 개로 나눈다.
- 다양한 hyperparmeter로 training set을 학습시키고, validation set으로 검증한다. validation set에서 가장 성능이 좋았던 hyperparameter를 선택한다.
- 위에서 만든 classifier를 가지고 test set에서 마지막에 딱 한 번만 성능을 test한다.
- 여기서 결국 중요한 것은 classifier가 한 번도 보지 못한 데이터(test set)를 얼마나 잘 예측하는지가 분류기의 성능을 결정한다는 것임

2️⃣ **두 번째 방법, cross validation (교차 검증)**

![Untitled](https://user-images.githubusercontent.com/79077316/158099532-f0e532ef-4836-49c6-8518-02b8d99fb0a0.PNG)

- 첫 번째 방법과 마찬가지로 training set과 test set을 나누어 test set은 마지막에 성능을 테스트할 때만 딱 한 번 사용한다.
- training set을 여러 개로 나눈다. 예를 들어 위 예제에서는 5-fold cross validation 예제라 5개로 나눈다.
- 처음 4개의 fold에서 hyperparameter를 학습시키고 나머지 한 fold에서 알고리즘을 평가한다.
- 위 과정을 모든 fold에 대해서 반복하여 최적의 hyperparameter를 찾는다.
- 이 방법은 데이터셋이 작을 때만 사용하여 딥러닝에서는 많이 사용하지 않음

### 이미지 분류기로써의 knn

- 실제로 입력이 이미지인 경우에는 knn을 잘 사용하지 않음
- 이유는
    1. test time에서 매우 느림
    2. L1/L2 distance가 이미지를 비교하는 척도로서 적절하지 않음
    3. 차원이 증가함에 따라 필요한 학습데이터의 양이 기하급수적으로 증가함

## 4. Linear classification

- Linear classification은 간단한 알고리즘이지만 neural network와 cnn의 기반이 되는 알고리즘으로 매우 중요함
- 강의에서는 neural network를 레고 블럭, Linear classifier를 낱개 블럭, cnn을 블럭으로 만든 거대한 타워에 비유함

### Parametric approach

- Linear classification은 parametric의 가장 단순한 형태임 (앞서 knn은 parameter가 없었다)

![Untitled](https://user-images.githubusercontent.com/79077316/158099544-1ae233c2-c239-49c5-93c4-7918670ffba0.PNG)

- parametric approach는 training set의 정보를 파라미터 w(가중치)에 요약함
- 따라서 test time 시 training data에 대한 정보가 필요 없게 됨. 파라미터 w만 있으면 됨
    - 앞서 knn에서는 test time에서 모든 학습 데이터에 대한 정보가 필요했고, 때문에 test time에서 시간이 많이 걸림
- 어떤 식으로 가중치 w와 데이터를 조합할지는 여러 가지 방법이 존재하고, 이게 모두 다양한 neural network 아키텍처를 설계하는 과정이다
- bias는 데이터와 무관하게 특정 클래스에 우선권을 부여하는 역할을 함
- Linear classifier는 곧 선형 1차 함수로, 분류하기 어려운 케이스도 많이 있음