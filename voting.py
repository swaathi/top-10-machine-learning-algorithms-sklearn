from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from base import Base

df = Base.clean()
X = df.drop(['Survived'], axis=1)
y = df['Survived']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


model_1 = LogisticRegression()
model_1.fit(Xtrain, ytrain)
ypred = model_1.predict(Xtest)

model_2 = GaussianNB()
model_2.fit(Xtrain, ytrain)
ypred = model_2.predict(Xtest)

model_3 = RandomForestClassifier()
model_3.fit(Xtrain, ytrain)
ypred = model_3.predict(Xtest)

eclf = VotingClassifier(
  estimators=[('lr', model_1), ('rf', model_2), ('gnb', model_3)],
  voting='hard'
)

for clf, label in zip([model_1, model_2, model_3, eclf], ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'Ensemble']):
  scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
  print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
