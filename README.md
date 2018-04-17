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

## Preparing training data

The training data are formed from 3 dataframes.  dfgood are positive ground truth data. Each tuple is explicitly calculated to represent the 3 edges of a right angle triangle.  dfnoise is the dataframe that are perturbed from dfgood and dfrand are just random tuples that fill the normalized data range (which is hard coded to 1000 for the two short edges). The concatenated data set are scrambled and split into two halves: one for DNN training and one for testing.  

```python
# ------------------ FORM THE TRAINING SET -----------------------------------------
# clean up the original dataframe to contain only  the 3 sides of the triangle
# label equals one means positive example
dfgood = df.copy()
dfgood['label']=np.ones(dataset_size)
dfgood.label = dfgood.label.astype(int)
dfgood['type'] = 'RightTri'

# create new dataframe with the same two sides as the good one. 
# the hypothenous is pertubed with some guassian noise
dfnoise = dfgood.copy()  # use .copy to not affect the original data frame
dfnoise.h = dfnoise.h + np.random.normal(norm_mu,norm_sigma,len(df))
dfnoise.a = dfnoise.a + np.random.normal(norm_mu,norm_sigma,len(df))
dfnoise.b = dfnoise.b + np.random.normal(norm_mu,norm_sigma,len(df))
# indicate the label as negative example (zero)
dfnoise.label = np.zeros(dataset_size).astype(int)
dfnoise['type'] = 'noise'

# create another dataframe with another random tuple as negative example
a1 = np.random.random(dataset_size)
b1 = np.random.random(dataset_size)
h1 = np.random.random(dataset_size)
l = np.zeros(dataset_size)
dfrand = pd.DataFrame({"a": a1, "b": b1, "h":h1,  "label":l}, index=index)
dfrand.a = dfrand.a * Normalrange
dfrand.b = dfrand.b * Normalrange
dfrand.h = dfrand.h * Normalrange * 1.414  # adjust for the height spread
dfrand.a = dfrand.a.astype(int)
dfrand.b = dfrand.b.astype(int)
dfrand['type'] = 'rand'

# Concat all 3 (or 2) dataframes into a training set
if add_rand:
	df_train = pd.concat([dfgood,dfnoise, dfrand])  
else:
	df_train = pd.concat([dfgood,dfnoise])
df_train = df_train.sample(frac=1) 				# scramble all the rows
df_train = df_train.head(dataset_rawsize) 		# select the top half for training
df_test = df_train.tail(dataset_rawsize)  		# select the bottom half for testing

```


## Training dataset visualization

A 5000 data point training set looks like this below. The left hand diagram is the positive ground truth and the right hand diagram are the superimposed negative ground truth.  The perturbed dataset (dfnoise) is further separated to the upper set and lower set (red and blue dots respectively).  The upper set is slightly above the positive ground manifold and the lower set is slightly below the manifold. 

![image of 5000 points training set](https://github.com/dennylslee/Pythagorean-classification-DNN/blob/master/5000TrainingSetVis.png)

## Basic DNN 

A two layer simple DNN is constructed using Keras. The input dimension is obviously set to 3 for the tuple input format.  No dropout was used since no signficant overfitting was observed given the simplicity of the problem.


```python
# ------------------ PREPARE DATA FOR DNN INPUT -----------------------------------------
# split into input (X) and output (Y) variables
X = df_train[['a','b', 'h']].values 		#convert dataframe to np array for Keras
Y = df_train['label'].tolist() 				#convert pd series to np array for Keras

# create model
model = Sequential()
model.add(Dense(num_firstlayer, input_dim=3, kernel_initializer='uniform', activation='relu'))
model.add(Dense(num_secondlayer, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs = epochs, batch_size= batch_size, validation_split = train_test_split, verbose=2)

```

## Results from varying training size

The predition results are overlayed on top of the raw time series.  The orange colored line is the training set and the green colored line is the testing set. The presence of a larger amount of training data minimizes the error (root mean square error) as the LSTM can fit the model from a more representative sequence. 

20 percent for training:

![image of 20pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-20pct-training.png)

10 precent for training:

![image of 10pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-10pct-training.png)

2 percent for training:

![image of 2pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-2pct-training.png)

