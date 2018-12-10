# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:38:07 2018

@author: bhuwan.madhikarmi

"""

# Importing libraries
import pandas as pd
import numpy as np
import operator

#### Start of STEP 1
# Importing data 
data = pd.read_csv("C:/Users/bhuwan.madhikarmi/Documents/MyPythonFolder/iris.csv")
#### End of STEP 1

data.head()
print(len(data)) #it gives number of total rows in the dataset
#150
print(data.shape) #it gives the number of rows and columns of the dataset
#150,5

# Defining a function which calculates euclidean distance between two data points
#here "length" is the number of cloumns in the testInstance, so it is 4 not 5
#iterator runs from 0 to 3
def euclideanDistance(data1, data2, length):
    distance = 0
    
    for x in range(length):
        #print("inner iteration: ",x,": ",np.sqrt(np.square(data1[x] - data2[x])))
        distance += np.square(data1[x] - data2[x])
        #"distance" variable holds sum of distances of each feature between the test instance and ecah of the training instances
        #calculate the distance between each feature and sum them up
        #print("x=",x," data1=",(data1[x]))
    return np.sqrt(distance)

# Defining our KNN model
def knn(trainingSet, testInstance, k):
 
    distances = {}
    sorted_d = {}
 
    length = testInstance.shape[1]
    #print("test instance shape 1:" +str(length))
    #### Start of STEP 3
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        #print("outer ieration: ",x)
        
        #### Start of STEP 3.1
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        #print("dist",dist)

        distances[x] = dist[0]
        print("distances[",x,"]: ",distances[x])
        #### End of STEP 3.1
 
    #### Start of STEP 3.2
    # Sorting them on the basis of distance
    #sorting using "sorted" method with 2 parameters give us a tuple (index,value) in result "sorted_d"
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #### End of STEP 3.2
    print("sorted_d")
    print(sorted_d)
    print("end of sorted_d")
    
 
    neighbors = []
    
    #### Start of STEP 3.3
    # Extracting top k neighbors
    print("for k = "+ str(k)+", the neighbours are: ")
    for x in range(k):
        neighbors.append(sorted_d[x][0])
        print("start of sorted_d["+str(x)+"][0]")
        print(sorted_d[x][0])
    #### End of STEP 3.3
    classVotes = {}
    print("the length of neighbors: "+str(len(neighbors)))
    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        print("neighbors[",x,"]:",neighbors[x])
        response = trainingSet.iloc[neighbors[x]][-1] # index -1 refers to last element, -2 second last
        response2 = trainingSet.iloc[neighbors[x],-1] # index -1 refers to last element, -2 second last        
        
        print("trainingSet.iloc[neighbors[x]]: ")
        print(trainingSet.iloc[neighbors[x]])
        print("response: ")
        print(response)
        print("response2: ")
        print(response2)
        
        if response in classVotes:
            classVotes[response] += 1
            print(classVotes)
        else:
            classVotes[response] = 1
            print(classVotes)
    #### End of STEP 3.4

    #### Start of STEP 3.5
    print("classVotes")
    print(classVotes)
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)    

    #### End of STEP 3.5
    
# Creating a dummy testset
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)
#### Start of STEP 2
# Setting number of neighbors = 1
k = 1
#### End of STEP 2
# Running KNN model
result,neigh = knn(data, test, k)

# Predicted class
print(result)
#-> Iris-virginica
# Nearest neighbor
print(neigh)
#-> [141]
 

#Now we will try to alter the k values, and see how the prediction changes.

# Setting number of neighbors = 3 
k = 3 
# Running KNN model 
result,neigh = knn(data, test, k) 
# Predicted class 
print(result) 
#-> Iris-virginica
# 3 nearest neighbors
print(neigh)
#-> [141, 139, 120]

# Setting number of neighbors = 5
k = 5
# Running KNN model 
result,neigh = knn(data, test, k) 
# Predicted class 
print(result) 
#-> Iris-virginica
# 5 nearest neighbors
print(neigh)
#-> [141, 139, 120, 145, 144]
 

#Comparing our model with scikit-learn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data.iloc[:,0:4], data['Name'])

# Predicted class
print(neigh.predict(test))

#-> ['Iris-virginica']

# 3 nearest neighbors
print(neigh.kneighbors(test)[1])
#-> [[141 139 120]]    
"""
Reference : 
https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/
"""
