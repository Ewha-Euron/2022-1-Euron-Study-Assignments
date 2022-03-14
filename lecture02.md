### 1. `Image Classsification`
: input 이미지가 주어졌을때, 주어진 카테고리 레이블에 알맞게 분류하는가 ex) 고양이사진 -> 고양이 

#### Problem
- Semantic Gap : 컴퓨터가 보는 것, 즉 pixel values와 실제 대상과는 큰 차이가 있음

#### Challenges
- Viewpoint variation : 대상이 그대로일때, 카메라 구도가 조금만 바뀌어도 pixel은 모두 바뀐다. 
- Illumination : 조명
- Deformation : 포즈, 위치 등의 변화
- Occlusion : 대상의 일부만 보일 때 
- Background Clutter : 배경색과 겹쳐 보일 때 (보호색)
- Intraclass variation : 다른 모양, 사이즈, 색상, 나이,,등 클래스 내의 다양성이 존재

#### 시도한 방법 
: edges를 찾아서 corners를 찾고 ... 이 방법을 각 카테고리마다 반복

💥 문제점1. super brittle 

💥 문제점2. scalable approach가 아님. 종류 하나하나씩 다 해야함.

#### Data-Driven Approach 
: 머신러닝을 통해 이미지를 학습하여 모델을 만들고, 그 모델을 통해 에측 및 평가 

### 2. `Nearest Neighbor `
: train에서 데이터와 레이블을 모두 기억하고 predict에서 학습 이미지와 가장 유사한 것의 라벨로 예측한다. 


→ '유사'하다고 어떻게 비교하는가? 
: L1 거리를 사용하여 이미지 사이의 행렬의 거리를 계산

- 숫자들 사이의 크기가 비슷하다면 사진도 비슷할 것
- 사진의 숫자들을 모두 빼고 절댓값을 씌움 -> L1거리 
- 이 방법을 사진의 모든 픽셀에 똑같이 적용, 그 후 나온 수를 모두 더해주면 두 사진 사이의 다른 정도가 나옴


💥 문제점. train 시간은 빠르지만, test 시간이 오래걸림. 모든 사진들을 다 빼주면서 계산해야하기 때문에

### 3. `K-Nearest Neighbor`
: 주변 k개를 봤을 때 가장 비슷한 것이 정답
  (k=1일 때, Nearest Neighbor과 같음) 
  
#### Distance Metric 
: L1(Manhatten distance) - 다이아몬드 
  L2(Euclidean distance) - 원 
  
#### Hyperparameters
: k, distance 처럼 직접 정해야하는 값. 초모수 
- train/validation/test로 나눔 
- train 데이터 셋에서 훈련된 값을 validation 데이터 셋으로 확인하며 하이퍼파라미터를 바꿔주고 test로 테스트하기 
- 🔺 데이터셋, 문제 마다 모두 다름. 

#### Cross-Validation 
: fold로 데이터를 나누어서 돌아가면서 validation으로 놓고 평균값으로 최종 하이퍼파라미터를 설정함 

#### 실무적으론..
→ Nearest Neighbor 와 K-Nearest Neighbor는 사용되지 않음 
- predict 과정이 굉장히 오래 걸림 
- 차원이 넓어질수록 필요한 데이터 수가 굉장히 많아짐 

### 4. `Linear Classification`
- 이미지 숫자들 (32*32*3), weight → 10개 카테고리에 대한 점수 
- 💥문제점. 카테고리마다 하나의 결과밖에 내지 못함. 
  

  
