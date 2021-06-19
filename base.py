import pandas as pd
from sklearn import metrics

from sklearn.model_selection import train_test_split

class Base:
  @staticmethod
  def accuracy_score(ytest, ypred):
    return metrics.accuracy_score(ytest, ypred).round(2)*100

  @staticmethod
  def clean_and_split():
    base = Base()

    # clean
    base.drop_useless_columns()
    base.one_hot_encode_columns()
    base.fill_in_nan()

    # split
    return base.split()

  @staticmethod
  def clean():
    base = Base()

    # clean
    base.drop_useless_columns()
    base.one_hot_encode_columns()
    base.fill_in_nan()

    return base.df

  def __init__(self, path='titanic/train.csv'):
    self.df = pd.read_csv(path)

  def drop_useless_columns(self):
    self.df.drop(['PassengerId'], axis=1, inplace=True)
    self.df.drop(['Name'], axis=1, inplace=True)
    self.df.drop(['Ticket'], axis=1, inplace=True)
    self.df.drop(['Cabin'], axis=1, inplace=True)

  def one_hot_encode_columns(self):
    # Encode Pclass
    dummies = pd.get_dummies(self.df['Pclass'], prefix='route')
    self.df = pd.concat([self.df, dummies], axis=1)
    self.df.drop(['Pclass'], axis=1, inplace=True)

    # Encode Sex
    dummies = pd.get_dummies(self.df['Sex'], prefix='route')
    self.df = pd.concat([self.df, dummies], axis=1)
    self.df.drop(['Sex'], axis=1, inplace=True)

    # Encode Embarked
    dummies = pd.get_dummies(self.df['Embarked'], prefix='route')
    self.df = pd.concat([self.df, dummies], axis=1)
    self.df.drop(['Embarked'], axis=1, inplace=True)

  def fill_in_nan(self):
    mean_age = self.df['Age'].mean()
    self.df['Age'].fillna(value=mean_age, inplace=True)

    # Checking if any column has NaN values
    # print(self.df.isna().any())

  def split(self):
    X = self.df.drop(['Survived'], axis=1)
    y = self.df['Survived']

    return train_test_split(X, y, random_state=0)
