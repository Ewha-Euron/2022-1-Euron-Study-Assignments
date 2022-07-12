# CS231n 1강 컴퓨터 비전의 역사💻👓

## 0. Intro

- 무수한 센서(ex. 카메라)로부터 비롯된 엄청나게 많은 시각 데이터
    
    → 이에 따라 visual data를 잘 활용할 수 있는 알고리즘의 중요성 증대
    
    → 하지만 실제로 이들을 이해하고 해석하는 일은 상당히 어려운 일임
    
- 매분 약 500시간 이상의 동영상이 업로드 됨 (2020년 유튜브 통계자료)

## 1. Computer Vision

- 컴퓨터 비전은 학제적(interdisciplinary)인 분야로, 굉장히 다양한 분야와 맞닿아 있다.
    - 물리학 - 광학, 이미지 구성, 이미지의 물리적 형성
    - 생물학, 심리학 - 동물의 뇌가 어떤 방식으로 시각 정보를 보고 처리하는지를 이해
    - 컴퓨터 과학, 수학, 공학 - 컴퓨터 비전 알고리즘을 구현할 컴퓨터 시스템을 구축할 때 필요

## 2. Vision의 역사

### 1) Biological Vision (**Evolution’s Big Bang)**

- 5억 4천만 년 전에 천만 년이라는 짧은 시간동안 생물의 종이 폭발적으로 증가한 시기가 있었음
    
    → 가장 설득력 있는 가설은 Biological vision의 탄생 (by. Andrew Parker)
    
- 5억 4천만 년 전 최초의 눈이 생겨나고 가장 중요한 감각기관으로 자리매김함

### 2) Mechanical Vision (camera)

- 1600년대 초창기의 카메라, Obscura
    - 핀홀 카메라 이론을 기반으로 함 - 빛을 모아주는 구멍 하나, 카메라 뒤편의 평평한 면에서 정보를 모으고 이미지를 투영시킴
    - 생물학적으로 발전한 초기의 눈과 유사

### 3) Hubel & Wiesel의 연구 (1959)

- “포유류의 시각적 처리 메커니즘은 무엇일까?”하는 질문으로부터 연구 시작
- 시각 처리 관점에서 인간의 뇌와 비슷한 고양이 뇌를 연구
    - 고양이 뇌 뒷면(primary visual cortex area가 있는)에 전극을 꽂아 어떤 자극을 줘야 뉴런들이 반응하는지를 관찰
- 연구로부터 시각 처리가 처음에는 단순한 구조로 시작하고, 그 정보가 통로를 거치면서 점점 복잡해진다는 것을 발견

## 3. Computer Vision의 역사

### 1) Block World (1963)

- 컴퓨터 비전에서 최초의 박사 학위 논문으로, 눈에 보이는 사물들을 기하학적인 모양으로 단순화시킴
- 연구의 목표는 눈에 보이는 세상을 인식하고, 그 모양을 재구성하는 것이었음

### 2) “The summer vision project” (1966)

- 컴퓨터 비전의 시작이 된 MIT의 프로젝트
- ‘The summer vision project is an attempt to use our summer workers effectively in the construction of a significant part of a virtual system.’

### 3) Vision(David Marr) (1970s)

- David Marr이 생각하는 비전이 무엇인지, 컴퓨터 비전이 어떤 방향으로 나아가야 하는지, 컴퓨터가 visual world를 인식하기 위해 어떤 알고리즘을 개발해야하는지를 담고 있는 책
- 우리가 눈으로 받아들인 이미지를 최종적인 full 3D 표현으로 만들려면 몇 단계의 과정을 거쳐야 한다고 주장
    
    ![Untitled](https://user-images.githubusercontent.com/79077316/156922509-807740ee-68e0-4ba3-ba6b-8c644626ba8b.png)
    
    1. **Input Image**
    2. **Primal Sketch** - 경계(edges), 막대(bars), 끝(ends), virtual lines, curves, boundaries가 표현
        
        이전에 Hubel과 Wiesel은 시각 처리의 초기 단계는 경계(edges)와 같은 단순한 구조와 밀접한 관련이 있다고 주장했음
        
    3. **2.5D Sketch** - 표면(surfaces), 깊이(depth), 층(layer), 불연속 점과 같은 것들을 종합하여 표현
    4. **3D Model** - 앞의 단계들을 모두 모아 surface and volumetric primitives 형태의 계층적으로 조직화된 최종적인 3D 모델
    
    → 이런 사고 방식은 오랫동안 비전에 대한 아주 이상적인, 지배적인 사고방식으로 여겨져 옴
    

### 4) Generalized Cylinder, Pictorial Structure (1970s)

![Untitled](https://user-images.githubusercontent.com/79077316/156922506-80a84c97-d977-4b11-af1e-0a5698747d9c.png)

- Generalized Cylinder - 사람을 원통 모양으로 조합해서 만듦
- Pictorial Structure - 사람을 주요 부위와 관절로 표현함
- 두 방법 모두 단순한 모양과 기하학적인 구성을 이용해서 복잡한 객체를 단순화시킴

### 5) by David Lowe (1987)

- David Lowe는 어떻게 해야 실제 세계를 단순한 구조로 재구성/인식할 수 있을지를 고민함
- 선(lines), 경계(edges), 직선(straight lines) 그리고 이들의 조합을 이용해서 사물을 재구성함
- 60/70/80 년대는 컴퓨터 비전으로 어떤 일을 할 수 있을지를 고민한 시기이지만, 단순한 toy example에 불과했음

### 6) Object Segmentation - Normalized Cut (1997)

- 객체 인식(object recognition)이 어렵다면 객체 분할(object segmentation)을 먼저 하자!
    - 여기서 질문! 근데 객체 분할이 객체 인식보다 더 어려운 task 아닌가? 여기서 말하는 객체 분할은 라벨링을 하지 않고 픽셀을 grouping 하는 방법을 얘기하는 건가?
- 객체 분할(Image segmentation)은 이미지의 각 픽셀을 의미 있는 방향으로 grouping 하는 방법
- 20세기부터 인터넷과 디지털 카메라의 발전으로 사진의 질이 훨씬 좋아졌고, 더 좋은 실험 데이터가 많이 생겨남

### 7) Object recognition - SIFT (1990s~)

![Untitled](https://user-images.githubusercontent.com/79077316/156922514-4f894ac0-59bd-44ca-9948-3925c45e8fff.png)

- 1990년대 후반 ~ 2010년대까지 특징(Feature)기반 객체 인식 알고리즘
- David Lowe의 **SIFT feature**를 사용하는 방법은 객체의 여러 특징 중 다양한 변화에 잘 변하지 않는 객체에서 중요한(diagnostic) 특징들을 찾아내고, 다른 객체에 그 특징들을 매칭시키는 방법

### 8) Face Detection - by Viola & Jones (2001)

- AdaBoost 알고리즘을 이용한 real-time face detection
- 컴퓨터 비전에서 유난히 발전 속도가 빨랐던 분야가 얼굴 인식
- 이 연구를 계기로 statistical machine learning 방법이 탄력을 받기 시작함
    - ex) support vector machine, boosting, graphical models, neural networks
- 2006년 Fuji Flim은 실시간 얼굴 인식이 가능한 최초의 디지털 카메라를 선보임

### 9) Support Vector Machine - Spatial Pyramid Matching (2006)

- 기본 아이디어는 이미지에서 특징을 잘 뽑아내면, 특징들이 이미지에 대한 단서를 제공해줄 것이라는 생각
- 이미지의 여러 부분과 여러 해상도에서 추출한 특징을 하나의 feature descriptor로 표현하고 Support vector machine 알고리즘을 적용함

### 10) Human Recognition - HoG, Deformable Part Model (2005, 2009)

![Untitled](https://user-images.githubusercontent.com/79077316/156922513-b56dceb9-710d-415c-a721-b151e8cf5f18.png)

### 11) PASCAL Visual Object Challenge (2006~2012)

- Benchmark Dataset 중 하나로, 20개의 object 클래스와 클래스당 수만 개의 이미지셋
- 위 데이터셋을 이용해 객체 인식(object recognition) 기술을 test한 결과 성능이 꾸준히 증가함

### 12) ILSVRC (2010~)

- 대부분의 머신러닝 알고리즘에서 overfitting의 문제 발생
    
    → 문제의 원인 중 하나는 시각 데이터가 너무 복잡하다는 것
    
    → 모델의 input은 복잡한 고차원의 데이터이고, 이로 인해 모델을 데이터셋에 fit하게 하려면 더 많은 parameters가 필요
    
    → 또 학습 데이터가 부족하면 overfitting이 훨씬 더 빠르게 발생했고, 일반화 능력이 떨어졌음
    
- ‘머신러닝의 overfitting 문제를 극복하면서, 세상의 모든 객체들을 인식할 수 있는가?’와 같은 물음으로 부터 ImageNet의 프로젝트가 시작됨
- ImageNet의 데이터는 약 15만 장의 이미지와 22만 가지의 클래스
- 2009년부터 ImageNet 팀이 ILSVRC 대회 주최
- 이 대회의 목적은 이미지 분류 문제를 푸는 알고리즘을 테스트하기 위함
- 2012년 챌린지에서 우승한 알고리즘이 바로 **Convolutional Neural Network(Deep Learning)**
    - CNN이 바로 컴퓨터 비전의 비약적인 발전을 이끌어낸 주역
    - 앞으로 이 강의에서는 CNN에 대해 중점적으로 다룰 것임

## 3. CNN

- 2012년에 CNN의 시대가 시작되었고, 이후 CNN 모델을 개선하고 튜닝하려는 많은 시도가 있었음
- CNN이 2012년에 처음 나온 것은 아니고, 오래전부터 존재했음

### 1) 2012년 이전의 CNN

**LeNet (1998)**

- 숫자 인식을 위한 CNN 모델 개발
- 이미지를 input으로 받아서 숫자와 문자를 인식할 수 있도록 함
- raw pixel을 입력으로 받아 여러 convolutional layer를 거치고 sub sampling, fully-connected layer를 거치게 됨
- 2012년의 많은 CNN 아키텍처들이 90년대의 LeNet 아키텍처를 기반으로 하기 때문에 아키텍처가 서로 비슷함

### 2) 2012년 이후의 CNN

- 2012년이 되어서야 CNN이 빛을 보게 된 이유는 바로 (1)**연산량의 증가**와 (2)**데이터의 증가** 때문임

(1) 연산량의 증가

- 연산량의 증가는 딥러닝 역사에서 아주 중요한 요소임
- 컴퓨터의 계산속도가 매년 빨라져서 계산 능력이 좋아짐 (무어의 법칙)
- GPU의 발전으로 강력한 병렬처리가 가능해짐, 이는 계산 집약적인 CNN 모델을 고속으로 처리하는데 적합함

(2) 데이터의 증가

- 90년대와 비교했을 때 사용 가능한 데이터셋이 매우 많아짐