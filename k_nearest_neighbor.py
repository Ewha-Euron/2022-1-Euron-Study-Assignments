from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self): #초기화 안함
        pass

    def train(self, X, y): #train 함수 
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X #데이터
        self.y_train = y #라벨

    def predict(self, X, k=1, num_loops=0): #predict 함수,L2를 구하는 함수를 호출하여 거리를 구하고 예측된 라벨 리턴하는 함수
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        
        #Distance를 루프가 없는 방식/루프 1개/루프2개로 구현함 
        if num_loops == 0:  
            dists = self.compute_distances_no_loops(X)  #루프 없을때 거리 
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X) #루프 1개
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X) #루프 2개 
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops) #그외에는 하지 않음

        return self.predict_labels(dists, k=k) #계산된 distance와 k-nn의 정해진 k값으로 예측한 라벨을 리턴함 

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0] #test point 갯수 =500
        num_train = self.X_train.shape[0] #train point 갯수  = 5000
        dists = np.zeros((num_test, num_train)) #여기에 test-train 간 거리 저장 
        
        #루프 2개 로 L2 distance를 구함
        #각각의 test point가 train point 5000개에 대하여 L2 거리를 구함 
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                #제곱의 루트, dist[i,j]는 test point i 번째와 train point j번째의 L2 거리  
                dists[i,j] = np.sqrt(np.sum(np.square(X[i]-self.X_train[j]))) 

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists 

    #루프 1개로 L2 distance를 구함
    def compute_distances_one_loop(self, X): 
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        for i in range(num_test): #모든 test point에 대하여 L2 구함 
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
            #dist[i, :]는 np array dist의 i번째 행에 행을 저장
            #X_train은 5000xD(차원) 
            #X[i]에는 i번쨰 test point가 있음,각각 1xD(차원)
            #차원이 같아서 broadcasting 으로 모든 train point의 각 행에 X[i]를 뺼수 있음
            #self.X_train-X[i] 의 결과 5000xD 형태, 각 행은 square로 제곱이 되고,가로로 더한것을 루트하면 L2
            #dist[i]는 1x5000의 형태로 dist는 (500,5000) 
            dists[i] = np.sqrt(np.sum(np.square(self.X_train-X[i]),axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    #루프 없이 L2 distance를 구함
    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #np.transpose([np.sum(np.square(X), axis=1)]) 는 500x1형태(각 행에는 각 test point의 요소를 제곱한것을 합해놓은 값)
        #np.sum(np.square(self.X_train), axis=1) 는 5000x1 형태(각 행에는 각 train point의 요소를 제곱한것을 합해놓은 값)
        #-2*np.dot(X, self.X_train.T) =(500x5000)
        #a=train, b=test라면 위를 모두 더하면 (a-b)^2이다. 
        dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis=1) + np.transpose([np.sum(np.square(X), axis=1)]))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS  LINE)*****
        return dists

    def predict_labels(self, dists, k=1):  #k=1일때 라벨을 예측해서 리턴 (NN)
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0] #test point의 갯수 
        y_pred = np.zeros(num_test) #y_pred에 test point 각각에 대한 라벨을 저장할것임
        
        for i in range(num_test): #모든 test point에 대하여 라벨 정하기
        
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #argsort는 어레이를 오름차순으로 정렬하여 바뀐 인덱스를 리턴함
            #self.y_train[np.argsort(dists[i])이 새롭게 오름차순으로 정렬된 y_train, 여기서 가장 거리 짧은 라벨 k개를 가져옴
            closest_y=self.y_train[np.argsort(dists[i])][:k]
            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #bincount()는 빈도수를 체크하는 함수, armax는 가장 큰값을 찾는 함수로 k=1이므로 가장 많이 라벨 결과로 있는 하나를 저장함 
            y_pred[i]=np.argmax(np.bincount(closest_y))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
