from sklearn.ensemble import RandomForestClassifier
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = RandomForestClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nRandom Forest Clasifier Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
