import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:/Users/anubh/Downloads/credit_scoring - credit_scoring.csv")

num_features = ['Age', 'Credit Utilization Ratio', 'Payment History',
                 'Number of Credit Accounts', 'Loan Amount',
                   'Interest Rate', 'Loan Term']
cat_features =['Gender', 'Marital Status', 'Education Level',
                'Employment Status', 'Type of Loan']

x=df.drop(columns='Type of Loan')
y=df['Type of Loan']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

ct = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first',sparse_output=False), cat_features)
],remainder='passthrough')

x_train_transformed = ct.fit_transform(x_train)
x_test_transformed = ct.transform(x_test)

print(x_train_transformed.shape)
print(x_test_transformed.shape)