from mnist import MNIST
from sklearn.linear_model import LogisticRegression
from time import time
import numpy as np

# Load mnist datasets
start = time()
mndata = MNIST('mnist-data')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
end = time()
print("load completed in {} seconds".format(int(end - start)))

# truncate set
# train_images = train_images[:100]
# train_labels = train_labels[:100]
# test_images = test_images[:]
# test_labels = test_labels[:]

# Train logistic regression classifier
logreg = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=100000, n_jobs=-1)
start = time()
logreg = logreg.fit(train_images, train_labels)
end = time()
print("train completed in {} seconds".format(int(end - start)))

# Check accuracy
accuracy = logreg.score(test_images, list(test_labels))
print("accuracy on test set is {} percent".format(accuracy * 100))

# save coefficients
np.savetxt("mat_coefficient.csv", logreg.coef_, delimiter=",")
np.savetxt("mat_intercept.csv", logreg.intercept_, delimiter=",")
np.savetxt("mat_classes.csv", logreg.classes_, delimiter=",")