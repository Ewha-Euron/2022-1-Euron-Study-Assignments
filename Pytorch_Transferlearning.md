# 컴퓨터 비전을 위한 전이학습(transfer learning) PyTorch 실습

전이학습을 이용하여 이미지 분류를 위한 CNN을 어떻게 학습시키는지 알아보겠다. 전이학습에서 중요한 2가지는 다음과 같다.

- **Finetuning the convnet**: 무작위 초기화 대신, 신경망을 ImageNet 1000 데이터셋 등으로 미리 학습한 신경망으로 초기화한다. 학습의 나머지 과정들은 평상시와 같다.
- **ConvNet as fixed feature extractor**: 여기서는 마지막에 fully connected layer를 제외한 모든 신경망의 가중치를 고정시킨다. 이 마지막 fully connected layer는 새로운 무작위의 가중치를 갖는 계층으로 대체되고, 이 레이어만 학습된다.

## 0. Import modules

필요한 모듈들을 다음과 같이 임포트해준다.

```python
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode
```

## 1. **Load Data**

데이터를 로드하기 위해 `torchvision` 과 `torch.utils.data` 패키지를 사용한다. 이번 실습에서는 개미와 벌을 분류하는 이미지 분류 모델을 훈련시키는 것이다. training set은 약 120장 정도 있고, test set은 75장 있다. 일반적으로 이 정도의 데이터셋이면 모델을 학습시키기에 매우 작은 데이터셋이지만, 우리는 여기서 **전이학습**을 할 것이므로 비교적 잘 모델을 일반화시킬 수 있다. 

```python
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'drive/MyDrive/Colab Notebooks/data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Data augmentation을 이해하기 위해 training 이미지 몇 개를 시각화해보겠다.

```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.

# 학습 데이터의 배치를 얻습니다.
inputs, classes = next(iter(dataloaders['train']))

# 배치로부터 격자 형태의 이미지를 만듭니다.
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

![Untitled](https://user-images.githubusercontent.com/79077316/166427284-9ae7bb66-94de-495a-9c91-94d3a8e2f06c.png)

## 2. **Training the model**

이제 모델을 학습시키기 위한 함수들을 작성해보겠다. 

- **Scheduling the learning rate**
- **Saving the best model**

아래에서 매개변수 `scheduler` 는 LR scheduler object이다.

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

## 3. **Visualizing the model predictions**

```python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```

## 4. **Finetuning the convnet**

미리 학습시킨 모델을 불러온 후 마지막에 fully connected layer를 초기화시킨다.

```python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

## 5. **Train and evaluate**

```python
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

![Untitled](https://user-images.githubusercontent.com/79077316/166427289-5d7bc940-a33a-4b70-a06e-41b4e8baf6b5.png)

```python
visualize_model(model_ft)
```

![Untitled](https://user-images.githubusercontent.com/79077316/166427296-b868fd9f-c503-477b-9711-2d1bd7fda73c.png)

## 6. **ConvNet as fixed feature extractor**

마지막 FC layer를 제외하고는 신경망을 고정시킨다. **requires_grad = False**를 설정해서 매개변수를 고정하여 `backward()` 중 gradient가 계산되지 않도록 한다.

```python
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

## 7. **Train and evaluate**

CPU에서 학습 및 평가를 하는 경우 이전과 비교했을 때 약 절반 가량의 시간만이 소요된다. 대부분의 신경망에서 gradinet를 계산할 필요가 없기 때문이다. 하지만 forward는 당연히 계산이 필요하다.

```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

![Untitled](https://user-images.githubusercontent.com/79077316/166427298-9ba29e84-bf3e-46a1-8f0d-1c9f12e79a99.png)

```python
visualize_model(model_conv)

plt.ioff()
plt.show()
```

![Untitled](https://user-images.githubusercontent.com/79077316/166427304-5f282997-a847-4f0a-9717-52ed0a21a1b1.png)