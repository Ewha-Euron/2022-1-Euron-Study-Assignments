# Pytorch로 이미지 분류 실습하기

다음과 같은 단계로 이미지 분류를 실습할 수 있다.

1. CIFAR10 데이터셋을 로드하고 정규화 한다. `torchvision`
2. Convolutional Neural Network을 define한다.
3. Loss funcion을 define한다.
4. 학습 데이터를 이용해서 네트워크를 훈련시킨다.
5. 테스트 데이터를 이용해서 네트워크를 테스트한다.

## 1. **Load and normalize CIFAR10**

`torchvision` 을 이용해서 CIFAR10 데이터셋을 로드한다.

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

Torchvision 데이터 세트의 출력은 [0, 1] 범위의 PILImage 이미지이다. 정규화된 범위 [-1, 1]의 텐서로 변환한다.

> PILImage란 이미지 분석 및 처리를 쉽게 할 수 있는 라이브러리(python imaging library, PIL)이다. pillow 모듈이라고 하고, 다양한 이미지 파일 형식을 지원하는 이미지 프로세싱 라이브러리의 한 종류이다. 
    

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

## **2. Define a Convolutional Neural Network**

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

## **3. Define a Loss function and optimizer**

Cross-Entropy 손실함수와 SGD+momentum으로 optimizer를 사용한다.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

## **4. Train the network**

data iterator를 반복시켜서 네트워크를 훈련시켜 보겠다.

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

![Untitled](https://user-images.githubusercontent.com/79077316/166426837-1e9b7f30-5927-46da-8d4b-3bcc3ff53e83.png)

그리고 다음 코드를 이용해서 훈련된 모델을 저장한다.

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

## **5. Test the network on the test data**

신경망이 예측하는 class label을 ground-truth와 비교해서 신경망을 테스트할 것이다. 

전체 신경망을 테스트 하기 전에 테스트 셋을 한번 확인해보겠다.

```python
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
```

![Untitled](https://user-images.githubusercontent.com/79077316/166426845-68a48bab-7b7f-4195-9ed6-b45eb9180052.png)

아까 저장했던 모델을 다시 로드한다.

```python
net = Net()
net.load_state_dict(torch.load(PATH))
```

훈련된 신경망이 어떻게 예측하는지 확인해보겠다.

```python
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
```

![Untitled](https://user-images.githubusercontent.com/79077316/166426850-c45f57c2-3c5a-4c5d-a45f-c07f74dddba4.png)

이제 네트워크가 전체 데이터셋에서 어떤 결과를 보이는지 알아보겠다.

```python
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

![Untitled](https://user-images.githubusercontent.com/79077316/166426855-65d37f07-7108-4e4d-9dd5-bc4f4220d7f6.png)