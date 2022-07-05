# Image-to-Image Translation with Conditional Adversarial Networks

📌 __condition adversarial networks as a general-purpose solution to image-to-image translation problems__
<br>
➡ effective at synthesizing photos from label maps, reconstructing objects from dege maps, and colorizing images
<br>
➡ wide applicability & ease of adoption without the need for parameter tweaking
<br>
➡ can achieve reasonable results without hand-engineering mappting functions & loss functions

Conditional adversarial networks learn
  1. the mapping from input image to output image
  2. a loss function to train this mapping

<br>

__automatic image-to-image translation__  <br>
: the task of translating one possible representation of a scene into another, given sufficient training data

📌 `Goal` : to develop a common framework for all problems

---

# Bringing Old Photos Back to Life

기존 문제
1. the degradation in real photos is complex
2. the domain gap between synthetic images and real old photos makes the network fail to generalize

➡ 이 문제를 해결하기 위해 논문에서 주장하는 것 <br>
: __a novel triplet domain translation network by leveragnig real photos along with massive synthetic image pairs__

1. train two variational autoencoders(VAEs) to respectively transform and clean photos into two latent spaces
2. the translation between two latent spaces is learned with synthetic paired data

To address multiple degradations mixed in old photo
1. a global branch with a partial nonlocal block targeting to the struectured defects(scratches, dust spots)
2. a local branch targeting to the unstructured defects(noises, blurriness) <br>
➡ two braches fused in the latent space <br>
➡ improved capability to restore old photos from multiple defects 

---

# Denoising Diffusion Proboabilistic Models

📌 __high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics__

This paper presents progress in diffusion probabilistic models.
`diffusion probabilistic model`: a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time

Diffusion models의 장점
- straightforward to define
- efficient to train
- __capable of generating high quality samples__
