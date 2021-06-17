from sklearn.linear_model import SGDClassifier
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = SGDClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nStochastic Gradient Descent Classifier Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
