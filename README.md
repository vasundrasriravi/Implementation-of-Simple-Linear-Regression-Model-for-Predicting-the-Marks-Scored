# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and sklearn.
2. Calculate the values for the training data set
3. Calculate the values for the test data set.
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VASUNDRA SRI R
RegisterNumber: 212222230168

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

df.tail()

X,Y=df.iloc[:,:-1].values, df.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(Xtrain,Ytrain)
Ypred=reg.predict(Xtest)
print(Ypred)

plt.scatter(Xtrain,Ytrain,color="orange")
plt.plot(Xtrain,reg.predict(Xtrain),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(Xtest,Ytest,color="blue")
plt.plot(Xtest,reg.predict(Xtest),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Ytest,Ypred)
print("MSE= ",mse)

mae=mean_absolute_error(Ytest,Ypred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
### df.head()
![head](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/fa14b195-55e0-4b59-8c36-549de4e46562)

### df.tail()
![tail](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/95853607-49ba-45aa-bf07-e73a166d2a72)

### X and Y values
![x y values](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/a7a46d86-ab36-458f-b067-f4bbb35d189e)

![Screenshot 2024-02-27 134531](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/f946fee6-a3be-4537-a944-8d8fc0dcb111)

### Training and Testing Set
![Screenshot 2024-02-27 134808](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/e57887f4-1bb9-41f8-acf8-e6b0438b05ca)

![Screenshot 2024-02-27 134826](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/dcc6802a-88be-4742-9285-11d401f0e4ee)

### Values of MSE,MAE and RMSE
![Screenshot 2024-02-27 134839](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/eee56194-229b-4c5e-be70-d2b229816942)

![Screenshot 2024-02-27 134851](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/e2b194bd-28d2-4b01-975b-61bf806b7d36)

![Screenshot 2024-02-27 134903](https://github.com/vasundrasriravi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393983/575fa8e1-7ec3-41b2-8de4-179f25f7eb29)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
