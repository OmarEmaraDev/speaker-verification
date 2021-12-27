# Speaker Verification

A speaker verification binary classifier implemented using both neural networks
and support vector machine. A college assignment.

## Dataset

The dataset is procedurally generated based on the
[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).

## Dependencies

- Numpy.
- Scipy.
- Tensorflow.
- Scikit-learn.
- python_speech_features.
- TeX-Distribution. (Documentation)

## Run

Run both the neural network and SVM classifiers. The code will write the results
needed by the documentation.

```
python neuralNetwork.py
python supportVectorMachine.py
```

Compile the documentation using latexmk if available.

```
cd documentation/
latexmk
```
