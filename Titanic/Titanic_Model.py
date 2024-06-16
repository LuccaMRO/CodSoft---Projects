import pandas  as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score,StratifiedKFold, train_test_split,GridSearchCV

data_set = r'C:\Users\mrolu\OneDrive\Documents\CodSoft - Projects\Titanic\Titanic-Dataset.csv'
df = pd.read_csv(data_set)

# print(df)

test_df, train_df = train_test_split(df,test_size=0.2,random_state=42)

# print(train_df.info())
# print(train_df.describe(include=['O']))

print(train_df.groupby(['Pclass'], as_index=False)['Survived'].mean())

print(train_df.groupby(['Sex'], as_index=False)['Survived'].mean())

print(train_df.groupby(['SibSp'], as_index=False)['Survived'].mean())

print(train_df.groupby(['Parch'], as_index=False)['Survived'].mean())

train_df['Family_size'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family_size'] = test_df['SibSp'] + test_df['Parch'] + 1

print(train_df.groupby(['Family_size'], as_index=False)['Survived'].mean())

print((train_df['Family_size'] == 7).sum())

family_map = {1:'Alone',2:'Small',3:'Small',4:'Medium',5:'Medium',6:'Large',7:'Large',8:'Large'}
train_df['Family_Size_Grouped'] = train_df['Family_size'].map(family_map)
test_df['Family_Size_Grouped'] = test_df['Family_size'].map(family_map)

print(train_df.groupby(['Family_Size_Grouped'], as_index=False)['Survived'].mean())

print(train_df.groupby(['Embarked'], as_index=False)['Survived'].mean())

sns.displot(train_df, x='Age',col='Survived', binwidth=10, height=5)
plt.show()

train_df['Age_cut'] = pd.qcut(train_df['Age'],8)
test_df['Age_cut'] = pd.qcut(test_df['Age'],8)

print(train_df.groupby(['Age_cut'], as_index=False)['Survived'].mean())