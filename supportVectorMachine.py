import numpy
from sklearn.svm import SVC
from loader import loadDataSet
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve

###########
# Read Data
###########
features, labels = loadDataSet(0)
features = features.reshape(features.shape[0], -1)

##################
# Cross Validation
##################
foldsCount = 5
rocCurves = []
accuracies = []
confusionMatrices = []
kFoldCrossValidator = KFold(foldsCount, shuffle = True, random_state = 0)
crossValidationSplits = kFoldCrossValidator.split(features, labels)
for trainingPermutation, validationPermutation in crossValidationSplits:
    trainingFeatures = features[trainingPermutation]
    validationFeatures = features[validationPermutation]
    trainingLabels = labels[trainingPermutation]
    validationLabels = labels[validationPermutation]

    ###########################
    # Construct And Train Model
    ###########################
    model = SVC(probability = True)
    model.fit(trainingFeatures, trainingLabels)


    ####################
    # Analyse And Record
    ####################
    accuracy = model.score(validationFeatures, validationLabels)
    accuracies.append(accuracy)

    predictionProbabilities = model.predict_proba(validationFeatures)[:, 1]
    rocCurve = roc_curve(validationLabels, predictionProbabilities)
    rocCurves.append(rocCurve)

    predictions = predictionProbabilities > 0.5
    confusionMatrix = confusion_matrix(validationLabels, predictions).ravel()
    confusionMatrices.append(confusionMatrix)

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
numpy.savetxt("documentation/supportVectorMachineROC.data",
        numpy.column_stack(rocCurve), comments = "",
        header = "falsePositiveRate truePositiveRate threshold")
