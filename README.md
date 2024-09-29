# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Padmavathi M
RegisterNumber:  212223040141
*/
```
```
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
## Array Value of x
![image](https://github.com/user-attachments/assets/d9d9e0ff-09fa-4fc1-87d3-cbc073d788a3)
## Array Value of y
![image](https://github.com/user-attachments/assets/b2885b17-e7cb-4c18-bb8e-fbce8920891f)
## Exam 1- score graph
![image](https://github.com/user-attachments/assets/07eeeb67-d9ab-4325-85af-df5efe774aa9)
## Sigmoid function graph
![image](https://github.com/user-attachments/assets/ce1b7d5b-873c-4a04-bf5d-ba3c7ee90dc6)
## X_train_grad value
![image](https://github.com/user-attachments/assets/eba4254b-4528-46ed-b8d1-e5d731d36ac7)
## Y_train_grad_value
![image](https://github.com/user-attachments/assets/1080a5ed-e78c-4b2d-b351-96f121843c29)
## Print res.x
![image](https://github.com/user-attachments/assets/48dd8a95-b627-40b1-b23c-a52df29d0cc8)
## Decision boundary - graph for exam score
![image](https://github.com/user-attachments/assets/be520732-d5af-43ee-880e-f9baa714703d)
## Probability value
![image](https://github.com/user-attachments/assets/013f9f7f-3477-40de-a966-38e90c3eb369)
## Prediction value of mean
![image](https://github.com/user-attachments/assets/f4f44c73-fd6c-433a-b9c7-efd694626d43)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

