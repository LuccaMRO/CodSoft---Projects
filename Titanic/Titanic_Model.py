import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV

data_set = r'C:\Users\mrolu\OneDrive\Documents\CodSoft---Projects\Titanic\Titanic-Dataset.csv'
df = pd.read_csv(data_set)

print(df)

test_df, train_df = train_test_split(df, test_size=0.2, random_state=42)

print(train_df.info())
print(train_df.describe())
print(train_df.describe(include=['O']))

print(train_df.groupby(['Pclass'], as_index=False)['Survived'].mean())

print(train_df.groupby(['Sex'], as_index=False)['Survived'].mean())

print(train_df.groupby(['SibSp'], as_index=False)['Survived'].mean())

print(train_df.groupby(['Parch'], as_index=False)['Survived'].mean())

train_df['Family_size'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family_size'] = test_df['SibSp'] + test_df['Parch'] + 1

print(train_df.groupby(['Family_size'], as_index=False)['Survived'].mean())

print((train_df['Family_size'] == 7).sum())

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Medium',
              5: 'Medium', 6: 'Large', 7: 'Large', 8: 'Large'}
train_df['Family_Size_Grouped'] = train_df['Family_size'].map(family_map)
test_df['Family_Size_Grouped'] = test_df['Family_size'].map(family_map)

print(train_df.groupby(['Family_Size_Grouped'],
      as_index=False)['Survived'].mean())

print(train_df.groupby(['Embarked'], as_index=False)['Survived'].mean())

print(sns.displot(data=train_df, x='Age', hue='Survived', multiple='stack', palette='Blues', fill=True))

g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 0) & (
    train_df['Age'].notnull())], color='Red', fill=True)
g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 1) & (
    train_df['Age'].notnull())], ax=g, color='Blue', fill=True)
g.set_xlabel('Age')
g.set_ylabel('Frequency')
g = g.legend(['Not Survived', 'Survived'])
plt.xlim(0, 71)
plt.show()

train_df['Age_cut'] = pd.qcut(train_df['Age'], 8)
test_df['Age_cut'] = pd.qcut(test_df['Age'], 8)

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

g = sns.kdeplot(train_df['Fare'][(train_df['Survived'] == 0) & (
    train_df['Fare'].notnull())], color='Red', fill=True)
g = sns.kdeplot(train_df['Fare'][(train_df['Survived'] == 1) & (
    train_df['Fare'].notnull())], ax=g, color='Blue', fill=True)
g.set_xlabel('Fare')
g.set_ylabel('Frequency')
g = g.legend(['Not survived', 'Survived'])
plt.xlim(0, 262.375)
plt.show()

train_df['Fare_cut'] = pd.qcut(train_df['Fare'], 6)
test_df['Fare_cut'] = pd.qcut(test_df['Fare'], 6)

print(train_df.groupby(['Fare_cut'], as_index=False)['Survived'].mean())

train_df.loc[train_df['Fare'] <= 7.775, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.775) & (
    train_df['Fare'] <= 8.662), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 8.662) & (
    train_df['Fare'] <= 14.454), 'Fare'] = 2
train_df.loc[(train_df['Fare'] > 14.454) & (
    train_df['Fare'] <= 26), 'Fare'] = 3
train_df.loc[(train_df['Fare'] > 26) & (train_df['Fare'] <= 52), 'Fare'] = 4
train_df.loc[train_df['Fare'] > 52, 'Fare'] = 5

test_df.loc[test_df['Fare'] <= 7.775, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.775) & (test_df['Fare'] <= 8.662), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 8.662) & (
    test_df['Fare'] <= 14.454), 'Fare'] = 2
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 26), 'Fare'] = 3
test_df.loc[(test_df['Fare'] > 26) & (test_df['Fare'] <= 52), 'Fare'] = 4
test_df.loc[test_df['Fare'] > 52, 'Fare'] = 5

train_df['Title'] = train_df['Name'].str.split(pat=',', expand=True)[1].str.split(
    pat='.', expand=True)[0].apply(lambda x: x.strip())
test_df['Title'] = test_df['Name'].str.split(pat=',', expand=True)[1].str.split(
    pat='.', expand=True)[0].apply(lambda x: x.strip())

train_df['Title'] = train_df['Title'].replace(
    ['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs')
train_df['Title'] = train_df['Title'].replace(
    ['Master', 'Dr', 'Rev', 'Don', 'Jonkheer', 'Sir'], 'Mr')

test_df['Title'] = test_df['Title'].replace(
    ['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
test_df['Title'] = test_df['Title'].replace(
    ['Master', 'Dr', 'Rev', 'Don', 'Jonkheer', 'Sir'], 'Mr')

print(train_df.groupby(['Title'], as_index=False)
      ['Survived'].agg(['count', 'mean']))


train_df['TicketNumber'] = train_df['Ticket'].apply(lambda x: x.split()[-1])
test_df['TicketNumber'] = test_df['Ticket'].apply(lambda x: x.split()[-1])

train_df['TicketnumberCount'] = train_df.groupby(
    'TicketNumber')['TicketNumber'].transform('count')
test_df['TicketnumberCount'] = test_df.groupby(
    'TicketNumber')['TicketNumber'].transform('count')

result = train_df.groupby(['TicketnumberCount'], as_index=False)['Survived'].agg(
    ['count', 'mean']).sort_values(by='count', ascending=False)
print(result)

print(train_df['Ticket'].str.split(pat=' ', expand=True))

train_df['TicketLocation'] = np.where(train_df['Ticket'].str.split(pat=' ', expand=True)[1].notna(
), train_df['Ticket'].str.split(pat=' ', expand=True)[0].apply(lambda x: x.strip()), 'NoLocation')
test_df['TicketLocation'] = np.where(test_df['Ticket'].str.split(pat=' ', expand=True)[1].notna(
), test_df['Ticket'].str.split(pat=' ', expand=True)[0].apply(lambda x: x.strip()), 'NoLocation')

print(train_df['TicketLocation'].value_counts())

print(train_df.groupby(['TicketLocation'], as_index=False)
      ['Survived'].agg(['count', 'mean']))


train_df['Cabin']= train_df['Cabin'].fillna('U')
train_df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train_df['Cabin']])

print(train_df.groupby(['Cabin'], as_index=False)
      ['Survived'].agg(['count', 'mean']))

train_df['Cabin_assigned'] = train_df['Cabin'].apply(lambda x: 0 if x == 'U' else 1)
test_df['Cabin_assigned'] = test_df['Cabin'].apply(lambda x: 0 if x == 'U' else 1)

print(train_df.groupby(['Cabin_assigned'], as_index=False)
      ['Survived'].agg(['count', 'mean']))