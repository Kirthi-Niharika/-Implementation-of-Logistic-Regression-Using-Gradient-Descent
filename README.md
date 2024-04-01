# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: T Kirthi Niharika
RegisterNumber: 212221040084
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
# Array Value of x:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/5a0db079-7565-4cd4-897c-841131521382)
# Array Value of y:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/4ab9d1c4-ea4d-4d99-869e-a34e084f5bf6)
# Exam 1 - score graph:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/469e842b-9964-4eb5-804f-57a745ac4acd)
# Sigmoid function graph:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/ece8e63a-31c1-4f6f-b636-a43efd116350)
# X_train_grad value:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/7fc8ea1e-96da-40a3-a7f3-3bb99d2b41b0)
# Y_train_grad value:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/f6af1db1-3956-442f-96cc-cd0e53e053e9)
# Print res.x:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/09c1b611-2714-46bf-879c-a36ed1bf7411)
# Decision boundary - graph for exam score:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/1a7fa1fd-7e69-4cb3-a3d4-c8d0d7e8955f)
# Probability value:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/d3af38b0-eabe-4bb7-96cf-2067bfce1306)
# Prediction value of mean:
![image](https://github.com/Kirthi-Niharika/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114135005/892e0425-fb00-4fe2-a9ab-13d11d39df08)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

