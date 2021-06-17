from sklearn.tree import DecisionTreeClassifier
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nDecision Tree Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
