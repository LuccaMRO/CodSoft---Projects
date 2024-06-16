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

data_set = r'C:\Users\mrolu\OneDrive\Documents\CodSoft---Projects\Titanic\Titanic-Dataset.csv'
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

train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 20), 'Age'] = 1
train_df.loc[(train_df['Age'] > 20) & (train_df['Age'] <= 24), 'Age'] = 2
train_df.loc[(train_df['Age'] > 24) & (train_df['Age'] <= 27), 'Age'] = 3
train_df.loc[(train_df['Age'] > 27) & (train_df['Age'] <= 34), 'Age'] = 4
train_df.loc[(train_df['Age'] > 34) & (train_df['Age'] <= 42), 'Age'] = 5
train_df.loc[(train_df['Age'] > 42) & (train_df['Age'] <= 51), 'Age'] = 6
train_df.loc[train_df['Age'] > 51, 'Age'] = 7

test_df.loc[test_df['Age'] <= 16, 'Age'] = 0
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 20), 'Age'] = 1
test_df.loc[(test_df['Age'] > 20) & (test_df['Age'] <= 24), 'Age'] = 2
test_df.loc[(test_df['Age'] > 24) & (test_df['Age'] <= 27), 'Age'] = 3
test_df.loc[(test_df['Age'] > 27) & (test_df['Age'] <= 34), 'Age'] = 4
test_df.loc[(test_df['Age'] > 34) & (test_df['Age'] <= 42), 'Age'] = 5
test_df.loc[(test_df['Age'] > 42) & (test_df['Age'] <= 51), 'Age'] = 6
test_df.loc[test_df['Age'] > 51, 'Age'] = 7

sns.displot(train_df, x='Fare',col='Survived', binwidth=10, height=5)
plt.show()

train_df['Fare_cut'] = pd.qcut(train_df['Fare'],6)
test_df['Fare_cut'] = pd.qcut(test_df['Fare'],6)

print(train_df.groupby(['Fare_cut'], as_index=False)['Survived'].mean())

train_df.loc[train_df['Fare'] <= 7.775, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.775) & (train_df['Fare'] <= 8.662), 'Fare'] = 1 
train_df.loc[(train_df['Fare'] > 8.662) & (train_df['Fare'] <= 14.454), 'Fare'] = 2
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 26), 'Fare'] = 3
train_df.loc[(train_df['Fare'] > 26) & (train_df['Fare'] <= 52), 'Fare'] = 4
train_df.loc[train_df['Fare'] > 52, 'Fare'] = 5

test_df.loc[test_df['Fare'] <= 7.775, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.775) & (test_df['Fare'] <= 8.662), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 8.662) & (test_df['Fare'] <= 14.454), 'Fare'] = 2
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 26), 'Fare'] = 3
test_df.loc[(test_df['Fare'] > 26) & (test_df['Fare'] <= 52), 'Fare'] = 4
test_df.loc[test_df['Fare'] > 52, 'Fare'] = 5

train_df['Title'] = train_df['Name'].str.split(pat=',',expand=True)[1].str.split(pat='.',expand=True)[0].apply(lambda x: x.strip())
print(train_df['Title'])