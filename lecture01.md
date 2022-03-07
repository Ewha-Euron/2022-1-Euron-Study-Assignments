## lecture 1. introduction - CV의 역사

- 1-7. 540million years, B.C. 이후 동물이 급격히 많아짐 -> evolution of vision의 중요성
- 1-8. 르네상스 1600s. camera obscura
- 1-9. 50s~60s - visual processing은 simple structure of visual world로 부터 시작됨. 
       electrophysiology : cat brain을 이용해서 simple cells, complex cells, ..
- 1-10. 1963년 CV의 시작. Larry Roberts
       original picture -> differentiated picture -> feature points selected
- 1-11. 1966년 
        The Summer Vision Project : an attempt to use our summer workers effectively in the construction of a significant part of a visual system. 
- 1-13. 1970s
        input image -> primal sketch -> 2 1/2-D sketch -> 3-D model representation
                       (edges, bars..)  (layers, depth,..)  (hierarchically organized in terms of surface..)
        
        Generalized Cylinder & Pictorial Structure
        기본 아이디어 : every object is composed of simple geomatic primitives.
- 1-16 ~ 1-20. 90s : object recognition이 어려우면, object segmentation 먼저 해보자!
        Image segmentation (1997)
        Face Detection (2001) - 컴퓨터가 엄청 느릴 때임 
        SIFT & Object recognition (1999) - 스탑 사인 매치하는 것 
        Spatial Pyramid Matching(2006) 
        * 많은 parts of images put them together -> feature descriptor -> support vectormachine algorithm
        Histogram of Gradients, Deformable Part Model (2005, 2009)
        
        => 60~80s보다 internet & digital camera의 발전, 그리고 data가 많아짐 
        
- PASCAL : 2007~2012까지 performance on defecting 20 objects 는 상승

- 기존 머신러닝 기법들은 오버피팅이 많은데 이미지는 특히 더욱 complex함.
- 또한 이미지가 충분하지 않기 때문에 오버피팅이 굉장히 빨리 나타남 
    => can't generalize 

- ImageNet의 goal
    1. to recognize object
    2. machine learning의 오버피팅을 극복하기 위해 
- 2012년 CNNs (Convolutional neural network) 등장으로 인해 image classification의 error rate가 약 10% 씩이나 줄어들게 됨

#### recognition image classification 
#### object detection
#### image captioning 

- 그런데 2012년 이전 1998년도에도 CNNs와 같은 기법, letter을 판단하기 위해 만들어진 알고리즘이 있었음. 
- CNNS의 key innovation
    1. 모어의 법칙으로 인해 computation up & GPUs (super parallelizable)
    2. data가 많아짐, 특히 많은 high quality labeled data set (PASCAL, ImageNet,..)
    
    
