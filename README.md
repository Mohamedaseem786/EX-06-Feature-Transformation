# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# Method For Data Tranformation

1. FUNCTION TRANSFORMATION
2. POWER TRANSFORMATION
3. POWER TRANSFORMATION
 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
### Data_To_Transform.csv:
~~~
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats 
df=pd.read_csv("Data_To_Transform.csv")  
df 
df.skew() 

#Log Transformation  
np.log(df["Highly Positive Skew"])  

#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])

#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])

#Square Transformation  
np.square(df["Highly Negative Skew"])

# POWER TRANSFORMATION
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df 
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df
~~~

# QUANTILE TRANSFORMATION:  
~~~
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show() 
df.skew()  
df 
~~~
###  titanic_dataset.csv:
~~~
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  
~~~
#FUNCTION TRANSFORMATION:
~~~  
#Log Transformation  
np.log(df["Fare"])  

#ReciprocalTransformation  
np.reciprocal(df["Age"])  

#Squareroot Transformation:  
np.sqrt(df["Embarked"])  

#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    

df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  

~~~
#QUANTILE TRANSFORMATION  
~~~
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  


df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  

sm.qqplot(df['Age_1'],line='45')  
plt.show()  

df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  

sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df  
~~~
# OUPUT
### Data_To_Transform.csv:
![](p1.png)
![](p2.png)
![](p3.png)
![](p4.png)
![](p5.png)
![](p6.png)
![](p7.png)
![](p8.png)
![](p9.png)
![](p10.png)
![](p11.png)
![](p12.png)
![](p13.png)
![](p14.png)
![](p15.png)
![](p16.png)
![](p17.png)
### titanic_dataset.csv:
![](p18.png)
![](p19.png)
![](p20.png)
![](p21.png)
![](p22.png)
![](p23.png)
![](p24.png)
![](p25.png)
![](p26.png)
![](p27.png)
![](p28.png)
![](p29.png)
![](p30.png)
![](p31.png)
![](p32.png)
![](p33.png)
![](p34.png)
![](p35.png)
# RESULT:
The various feature transformation techniques has been performed on the given datasets and the data are saved to a file.