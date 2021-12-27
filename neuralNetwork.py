import numpy
from loader import loadDataSet
from tensorflow.math import sigmoid
from tensorflow.keras import Sequential
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix, roc_curve

######################
# Global Configuration
######################
set_random_seed(0)

###########
# Utilities
###########

# The BinaryAccuracy metric assume a probabilistic output For binary
# classification, the recommendation, however, is to use the logits aware
# BinaryCrossentropy loss function and assign a linear activation function to
# the last layer, consequently this metrics can not be used. The following class
# reimplements the metric but with manual sigmoid application for use with
# binary classifiers.
class LogitsBinaryAccuracy(BinaryAccuracy):
    def update_state(self, reference, prediction, sample_weight = None):
        return super().update_state(reference, sigmoid(prediction), sample_weight)

###########
# Read Data
###########
features, labels = loadDataSet(0)

##################
# Cross Validation
##################
foldsCount = 5
rocCurves = []
accuracies = []
lossCurves = []
confusionMatrices = []
kFoldCrossValidator = KFold(foldsCount, shuffle = True, random_state = 0)
crossValidationSplits = kFoldCrossValidator.split(features, labels)
for trainingPermutation, validationPermutation in crossValidationSplits:
    clear_session()
    trainingFeatures = features[trainingPermutation]
    validationFeatures = features[validationPermutation]
    trainingLabels = labels[trainingPermutation]
    validationLabels = labels[validationPermutation]

    #################
    # Construct Model
    #################
    model = Sequential((
        Flatten(input_shape = trainingFeatures.shape[1:]),
        Dense(128, activation = relu),
        Dense(1)
    ))
    model.compile(loss = BinaryCrossentropy(from_logits = True),
        optimizer = Adam(), metrics = (LogitsBinaryAccuracy()))

    ####################
    # Train And Validate
    ####################
    earlyStopping = EarlyStopping(monitor = "val_binary_accuracy", mode = "max",
        patience = 3, restore_best_weights = True)
    history = model.fit(trainingFeatures, trainingLabels, epochs = 100,
        validation_data = (validationFeatures, validationLabels),
        callbacks = (earlyStopping))

    ####################
    # Analyse And Record
    ####################
    accuracy = numpy.max(history.history["val_binary_accuracy"])
    accuracies.append(accuracy)

    predictionProbabilities = sigmoid(model(validationFeatures))
    rocCurve = roc_curve(validationLabels, predictionProbabilities)
    rocCurves.append(rocCurve)

    predictions = predictionProbabilities > 0.5
    confusionMatrix = confusion_matrix(validationLabels, predictions).ravel()
    confusionMatrices.append(confusionMatrix)

    lossCurves.append(history.history["loss"])

################
# Output Results
################
print(f"Accuracy: {numpy.mean(accuracies)}")

# It is unclear how metrics like ROC curves and confusion matrices should be
# presented within cross validation. Averaging is not possible because the
# metrics can have different lengths and dimensions. So we present the metrics
# recorded for the cross validation fold that has the median validation
# accuracy.
medianIndex = numpy.argpartition(accuracies, foldsCount // 2)[foldsCount // 2]

confusionMatrix = confusionMatrices[medianIndex]
print(f"True Negative: {confusionMatrix[0]}")
print(f"False Positive: {confusionMatrix[1]}")
print(f"False Negative: {confusionMatrix[2]}")
print(f"True Positive: {confusionMatrix[3]}")

rocCurve = rocCurves[medianIndex]
numpy.savetxt("documentation/neuralNetworkROC.data", numpy.column_stack(rocCurve),
        comments = "", header = "falsePositiveRate truePositiveRate threshold")

lossCurve = lossCurves[medianIndex]
numpy.savetxt("documentation/neuralNetworkLoss.data", lossCurve, comments = "",
        header = "loss")
