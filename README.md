# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
``
```
1.Import the required packages.
2.Read the given dataset and assign x and y array.
3.Split x and y into training and test set.
4.Scale the x variables.
5.Fit the logistic regression for the training set to predict y.
6.Create the confusion matrix and find the accuracy score, recall sensitivity and specificity.
7.Plot the training set results.
```

    

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: sreenithi.E
RegisterNumber:  212223220109
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)
ypred=c.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(('skyblue','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(('black','white'))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


````
``

## Output:
```
```
Array of X:

![Screenshot 2024-04-02 231630](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/0a3368d6-126b-4e9e-8f14-fbe923356fd0)

Array of Y:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/db22a3e6-35e1-47a8-9aad-efe080577ac2)

Score Graph:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/47f56532-e201-4990-8412-1ff0c0e0698e)

Sigmoid Function Graph:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/ac3fb8e0-8a09-47c7-9f7f-2af01a8ef019)

X_train_grad Value:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/547a5086-31d5-4771-b193-59068c32e52a)

Y_train_grad Value:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/542da555-b6bf-4b1e-b5df-9aa775d70491)

Print res_X:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/3e4a5d91-f678-4bdb-9805-accc232a01a7)

Decision boundary:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/e33cb9e0-39ea-4bc3-9fcc-f901f05ec4a1)

Probability Value:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/bf90084c-dce2-45e7-9b91-0e833ba607b8)

Prediction Value of Mean:

![image](https://github.com/sreenithi123/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743046/f1b4d0c0-5fa3-49c8-a740-157c16b89668)









## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

