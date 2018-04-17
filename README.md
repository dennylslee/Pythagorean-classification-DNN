# Introduction

This project is to use a DNN to perform one-class classification (OCC) of tuples.  Each tuple represents the lengths of the 3 edges of a triangle.  The task is for OCC to correctly identified if the tuple is a right angle triangle, or not. In essence, this project is for DNN to approximate the Pythagorean theorem relationship. 

The OCC problem is set up as a typical supervised learning method.  Ground truth data set are fed into the DNN for training purposes. The training dataset includes:

1.  positive training data: these are tuple values that meet the pythagorean theorem.
2.  negative training data: using the positve training data as a base, each data point is perturbed by adding some Guassian noise.  The spread of this noise is controllable.  The narrower the spread, the harder it is for the DNN to classify correctly since the tuple value set is very close to the positive ground truth. 
3.  negative training data: these tuple values are simply random numbers representing all sorts of non right angle triangle. 

The main objective of this project is to:
a. understand if a reasonably small DNN can approximate the Pythagorean theorem and perform this OCC accurately.
b. understand the sensitivity to the type of training data (positive, negative, etc.)
c. understand the sensitivity to the amount of training data.
d. understand the sensitivity to the size of the DNN 

A good academic reference to the subject of optimal% of negative samples in performing classification is [here](https://www.researchgate.net/publication/229039329_Determining_the_optimal_percent_of_negative_examples_used_in_training_the_multilayer_perceptron_neural_networks)

This project is built using python and the Keras framework. 

## Time series generation

The raw time series is created 4 raw components (effectively modulation):
1. a so-called carrier cosine wave 
2. a low frequncy sine wave modulation
3. a slow rising parabola curve 
4. guassian noise added on top at each timestep

The main code construct:

```python
T
```

The raw waveform (time series) looks like this:

![image of raw ts](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-rawTS.png)

## LSTM prediction

A single layer LSTM is used for performing the prediction.  The internal state (vector) size of the cell and hidden state is set as 10. The look_back variable controls the size of the input vector into the RNN(LSTM).  

Sensitivity analysis options:
1. The training size proportion
2. Look back (i.e. the timestep of the input vector) 
3. LSTM unit which is the internal vector size of the cell (memory) and hidden state 

```python
# split into train and test sets
# control the proportion of training set here
train_size = int(len(dataset) * 0.02)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
# look_back dictates the time steps and the hidden layer; can cause overfitting error when it's too large
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
 
# reshape input to be [samples, time steps, features]
# NOTE: time steps and features are reversed from example given
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1],1))

# create and fit the LSTM network
# single hidden layer (timestep is one) and cell states
# The "unit" value in this case is the size of the cell state and the size of the hidden state
model = Sequential()
model.add(LSTM(10, input_shape=(look_back,1))) 	# NOTE: time steps and features are reversed from example given
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
```

## Results from varying training size

The predition results are overlayed on top of the raw time series.  The orange colored line is the training set and the green colored line is the testing set. The presence of a larger amount of training data minimizes the error (root mean square error) as the LSTM can fit the model from a more representative sequence. 

20 percent for training:

![image of 20pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-20pct-training.png)

10 precent for training:

![image of 10pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-10pct-training.png)

2 percent for training:

![image of 2pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-2pct-training.png)

