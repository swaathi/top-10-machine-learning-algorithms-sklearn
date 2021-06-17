from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = BaggingClassifier(
  base_estimator=SVC(),
  n_estimators=10,
  random_state=0
)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nBagging Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
