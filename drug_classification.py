"""
DRUG CLASSIFICATION
"""


#%% # IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

#%% Read the dataset

data = pd.read_csv("drug200.csv")


#%% EDA (Exploratory Data Analysis)

# Let's look at the columns
data.columns

# Variable description
"""
age = age of the patient
Sex = gender of the patient
BP = blood pressure levels
Cholosterol = Cholosterol levels
Na to K = Na to Potassium Ratio
Drug = type of drug (target columns)
"""

data.info()
data.describe

data.head()  # first 5 rows

# As I said that Drug is the target column , I'm going to change the column name for it.

data.rename({"Drug":"Target"},axis = 1,inplace = True) # inplace , which means change the name and implement on the dataset.

# Categorical Variables = (Sex,BP,Cholesterol,Na_to_K,Drug)

def bar_plot(variable):
    
    # get feature 
    var = data[variable]
    
    # count number of the categorical variables
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize =(5,5))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.xlabel("Frequncy")
    plt.title(variable)
    plt.show()
    print("{} : {}",variable,varValue)

categorical_variables = ["Sex","BP","Cholesterol","Target"]
for c in categorical_variables:
    bar_plot(c)
    
# Numerical Variables = (Na_to_K)

def plot_hist(numerical_variable):
    plt.figure(figsize = (5,5))
    plt.hist(data[numerical_variable],bins = 100,color = "green")
    plt.xlabel(numerical_variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(numerical_variable))
    plt.show()

numerical_variables = ["Na_to_K"]
for b in numerical_variables:
    plot_hist(numerical_variables)

#%% Missing Values

# To check which column has a missing value
data.columns[data.isnull().any()]
data.isnull().sum()

# Dataset hasn't missing value


#%% Outlier detection
def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers

data.loc[detect_outliers(data,["Age","Na_to_K"])]


#%% I'm going to convert categorical variables to numerical variables

from sklearn import preprocessing

# I created a object of th
label_encoder = preprocessing.LabelEncoder()

data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["BP"] = label_encoder.fit_transform(data["BP"])
data["Cholesterol"] = label_encoder.fit_transform(data["Cholesterol"])
data["Target"] = label_encoder.fit_transform(data["Target"])

#%% Get X and Y Coordinates

y = data.Target.values

x_data = data.drop(["Target"],axis = 1) # axis = 1 ---> column , axis = 0 ----> row

#%% Normalization Operation

x = (x_data-np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

#%% Train - Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42) # random state,which means when the machine divide the dataset as random , divide same ratio all the time.

#%% Correlation Matrix

corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot = True,fmt = ".2f",figsize=(10,10))
plt.title("Correlation Between Features(Columns)")
plt.show()

#%% Logistic Regression with sklearn
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Accuracy of The Logistic Regression : ",lr.score(x_test,y_test))

# Accuracy of The Logistic Regression :  0.775

#%% K-Nearst Neighbor Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
prediction = knn.predict(x)

print("For k value =  {} Accuracy of the K-Nearst Neighbor Classification : {}".format(1,knn.score(x_test,y_test)))
# For k value =  3 Accuracy of the K-Nearst Neighbor Classification : 0.9

#-----------------------------------------------#
# Let's find best k value for K-Nearst Neighbor Classification

score_list = []

for each in range(1,160):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,160), score_list)
plt.title("K-Value & Accuracy")
plt.xlabel("K-Value")
plt.ylabel("Accuracy")
plt.show()


"""
if we set 1 as a k value , ıt shows us to best accuracy
"""

"""
For k value =  1 Accuracy of the K-Nearst Neighbor Classification : 0.925
"""


#%% Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,random_state = 1) # n_estimators , which means number of trees
rf.fit(x_train,y_train)


print("Accuracy of Random Forest Classification : ",rf.score(x_test,y_test))


"""
Accuracy of Random Forest Classification :  1.0
"""

#%% 

y_pred = rf.predict(x_test)
y_true = y_test


#%% Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

#%% Confusion Matrix Visualize

import seaborn as sns
import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax =ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()