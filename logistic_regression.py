from sklearn.linear_model import LogisticRegression
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = LogisticRegression()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nLogistic Regression Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
