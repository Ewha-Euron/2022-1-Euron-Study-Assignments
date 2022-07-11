## Lec4 - Introduction to Neural Networks
---

### Backpropagation과 Neural Network


- 임의의 복잡한 함수를 통해 어떻게 analytic gradient를 계산할까?
- computational graph를 사용해서 함수를 표현하게 됨으로써 backpropagation이라고 부르는 기술을 사용할 수 있게 된다.

`backpropagation`

: gradient를 얻기위해 computational graph 내부의 모든 변수에 대해 chain rule을 재귀적으로 사용

`forward pass`

: 입력값을 받아서 loss값을 구하기까지 계산해 가는 과정

`backward pass`

: forward pass가 끝난 이후 역으로 미분해가며 기울기 값들을 구해가는 과정

#### Patterns in Backward Flow
![image](https://user-images.githubusercontent.com/63354176/160359745-b469e38e-9a5d-4bcb-a349-f5882a1f4a9a.png)

#### Modularized implementation
![image](https://user-images.githubusercontent.com/63354176/160360260-d03fb655-a2ca-4304-b80c-7241176d6bcf.png)

- forward pass : 인자 값은 x와 y, 리턴 값 z=x*y
- backward pass :  loss 값 L을 z로 미분한 값을 인자로, 리턴 값은 L을 x, y로 미분한 값

![image](https://user-images.githubusercontent.com/63354176/160360618-9a9b132d-f072-4f9d-8d4d-774717031bc2.png)
- dx = self.y * dz
- dy = self.x * dz

----
## Neural Networks
![image](https://user-images.githubusercontent.com/63354176/160360940-facb46f0-a1e8-44c1-8572-a773ccdfc4c0.png)

하나만 말고, 다른 W 레이어 두 개를 지나게 만들자! `f=W2*max(0, W1x)`

((뒤에 내용은 아직 이해가 잘 안돼서 나중에 추가하겠습니다))

레이어를 2개, 3개 ... 여러개를 씌우게 되는 것 = 인공 신경망
![image](https://user-images.githubusercontent.com/63354176/160361709-56c70eb1-760d-4755-898a-d0923d55c169.png)

`fully-connected layer`

: 중간중간에 모든 노드가 다음의 모든 노드에 영향을 끼치는 레이어
