import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

train_data=pd.read_csv('trainRegression.csv')
x=np.array(train_data['X'])
y=np.array(train_data['R'])

A=np.empty((2,2),dtype=int)
for i in range(2):
    for j in range(2):
        A[i,j]=np.sum(x**(i+j)) if (i+j)!=0 else len(x)
#print("Matrix A: ",A)

B=np.empty((2,1),dtype=int)
for i in range(2):
    B[i,0]=np.sum(y*x**i)
#print("Matrix B: ",B)

ans=np.linalg.solve(A,B)
print("Constants: ",ans)

predtrain_data=(ans[0,0])+(ans[1,0]*x)
print("Pred on train_data",predtrain_data)

MSE_traindata=(np.sum((predtrain_data-y)*(predtrain_data-y)))/len(x)
print("Mean Square Error: ",MSE_traindata)

plt.title("Training Data")
plt.plot(x,y,'.')
plt.xlabel("X-values")
plt.ylabel("R-values")
plt.plot(x,predtrain_data)
plt.show()

test_data=pd.read_csv("testRegression.csv")
x1=np.array(test_data['X'])
y1=np.array(test_data['R'])

predtest_data=(x1*ans[1,0])+ans[0,0]
print("Pred on Test_data: ",predtest_data)

MSE_testdata=(np.sum((predtest_data-y1)*(predtest_data-y1)))/len(x1)
print("Mean Square Error: ",MSE_testdata)

plt.title("Testing Data")
plt.plot(x1,y1,'.')
plt.xlabel("X-values")
plt.ylabel("R-values")
plt.plot(x1,predtest_data)
plt.show()

#Result:The model performs poorly as the predicted line does not fit the data closely.