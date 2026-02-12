# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.  Import required libraries and load the dataset.
2.  Drop unnecessary columns and convert categorical variables using get_dummies().
3.  Split the data into features (X) and target (Y).
4.  Apply StandardScaler to normalize the data.
5.  Split the dataset into training and testing sets.
6.  Create and train the SGD Regressor model.
7.  Predict the test data values.
8.  Calculate MSE, MAE, and R² score.
9.  Plot actual vs predicted values.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

#load the data set
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

#data preprocessing,dropping the unnecessary coloumn and handling the catergorical variables
data=data.drop(['CarName','car_ID'],axis=1)
data = pd.get_dummies(data, drop_first=True)

#splitting the data 
X=data.drop('price', axis=1)
Y=data['price']

scaler = StandardScaler()
X=scaler.fit_transform(X)
Y=scaler.fit_transform(np.array(Y).reshape(-1, 1))

#splitting the dataset into training and tests
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#create sdg regressor model
sgd_model= SGDRegressor(max_iter=1000, tol=1e-3)

#fiting the model to training data
sgd_model.fit(X_train, Y_train)

#making predictions
y_pred = sgd_model.predict(X_test)

#evaluating model performance
mse = mean_squared_error(Y_test, y_pred)
r2=r2_score(Y_test,y_pred)
mae= mean_absolute_error(Y_test, y_pred)

#print evaluation metrics
print('Name:Rosetta Jenifer C')
print('Reg no: 212225230230')
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-Squared Score:",r2)

#print model coefficients
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#visualising actual vs predicted prices
plt.scatter(Y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(Y_test),max(Y_test)],[min(Y_test),max(Y_test)],color='red')
plt.show()
```

## Output:
<img width="562" height="141" alt="image" src="https://github.com/user-attachments/assets/f47504b8-0188-428d-a7ff-3699e8aed453" />
<img width="992" height="302" alt="image" src="https://github.com/user-attachments/assets/365aea9d-adb8-4246-8725-e9a23e84aadd" />
<img width="1052" height="657" alt="image" src="https://github.com/user-attachments/assets/6443f934-7352-411c-8cff-4ad586791b3f" />

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
