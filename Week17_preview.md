#GAN, Image-to-Image Translation with Conditional Adversarial Networks 


- image to image translation에서 general-purpose solution 제시
- Conditional GAN을 이용한 한 유형의 이미즐 다른 유형의 이미지로 변환하는 framework 제시하여 image to image translation 작업에
  처음 적용해서 좋은 결과를 얻음 



#Bringing Old Photos Back to Life 

- 이미지  복원에서 화질을 복원하는 연구이고,Triplet domain translation을 제시하여 기존 이미지 복원 연구에서 가지고 있던 한계점인
  Generalization issue와 mixed degradation issue 에 대한 해결방안을 제시함.
- scratch가 이미지 전제적으로 일관적으로 복원될수 있게됨   

#Denoising Diffusion Probabilistic Models (DDPM) 

- nonequilibrium thermodynamics로부터 고안된 잠재 변수 모델 중 하나인 diffusion probabilistic models를 제세.
- forward diffusen process: noise를 점점 증가 시켜가면서 학습 데이터를 특정한(Guassian) noise distribution으로 변환
- reverse generative proccess: noise distribution으로부터 학습 데이터를 복원(denoising)하는 학습 단위과정을 Markov chain으로 표현
