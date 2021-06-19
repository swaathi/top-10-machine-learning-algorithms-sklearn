from xgboost.sklearn import XGBClassifier
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = XGBClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nXG Boost Classifier Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
