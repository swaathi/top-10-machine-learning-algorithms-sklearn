from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = GaussianNB()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nNaive Bayes Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")

mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.savefig("visualizations/naive_bayes_confusion_matrix.png")
