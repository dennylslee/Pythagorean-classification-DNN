import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# --------------------- Configuration ----------------------------------------
# training data control
dataset_rawsize = 5000
dataset_size =  dataset_rawsize * 2 # will split up half for testing later 
Normalrange  = 1000
add_rand = True # switch to add in the random negative sample set
# control how spread the noise is
norm_mu,norm_sigma = 0, 10
# DNN control 
num_firstlayer = 40
num_secondlayer = 20
epochs = 50
train_test_split = 0.25		# pct left for test validation. Keras use the latter port of dataset
batch_size = 10
# plot style control
style.use('ggplot')
plotmanifold = True

# ------------------ FORM THE RAW SET -----------------------------------------
# set up a random set of edges of triangle
a = np.random.random(dataset_size)  # first triangle edge
b = np.random.random(dataset_size)  # second triangle edge
index = np.arange(1,dataset_size+1,1)
df = pd.DataFrame({"a": a,  "b": b}, index=index)

# Normalize the random number and convert to integer
df['a']=df['a']*Normalrange
df['b']=df['b']*Normalrange
df.a = df['a'].astype(int)
df.b = df['b'].astype(int)
df['h'] = np.sqrt(df.a**2 + df.b**2)	# Do the Pythagorean theorem to build the ground truth

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

print('------------------------------------------------------------------')
print(df_train.head())
print(df_train.tail())
print('------------------------------------------------------------------')
print('Training dataset size is:', len(df_train))
print('Testing dataset size is:', len(df_test))
print('Perturbed negative samples spread (std deviation) is:', norm_sigma)
print('------------------------------------------------------------------')

# -------------------------------- PLOT THE MANIFOLD ---------------------------------
if plotmanifold:
	fig = plt.figure(figsize=(12,5))
	# fig.tight_layout(h_pad=0, w_pad=0)	
	plt.subplots_adjust(wspace = 0.1)
	# plot the positive samples
	clusterplot = fig.add_subplot(121,projection='3d')
	plt.title('Positive Training Data Visualization', loc='Left', weight='bold', color='Black')
	xplot = df.a.tolist()
	yplot = df.b.tolist()
	zplot = df.h.tolist()
	clusterplot.set_xlabel ('a',weight='bold')
	clusterplot.set_ylabel ('b',weight='bold')
	clusterplot.set_zlabel ('hypothenous',weight='bold')
	clusterplot.scatter(xplot, yplot, zplot, s=20, c='orange', edgecolors='grey', linewidths=1)
	# plot the whole training set
	clusterplot = fig.add_subplot(122,projection='3d')
	plt.title('Negative Training Data Visualization', loc='Left', weight='bold', color='Black')
	# split the negative pertubed samples to two lists
	# one above and one below the positive samples manifold; which allowed for different color plotting
	dfnoise['hactual'] = np.sqrt((dfnoise.a)**2 + (dfnoise.b)**2)
	dfnoise['high'] = np.where(dfnoise['hactual'] <= dfnoise['h'], True, False)
	dfnoisehi = dfnoise[dfnoise['high'] == True]
	dfnoiselow = dfnoise[dfnoise['high'] == False]
	xplot = dfnoisehi.a.tolist()
	yplot = dfnoisehi.b.tolist()
	zplot = dfnoisehi.h.tolist()
	clusterplot.scatter(xplot, yplot, zplot, s=5, c='red', linewidths=1)
	xplot = dfnoiselow.a.tolist()
	yplot = dfnoiselow.b.tolist()
	zplot = dfnoiselow.h.tolist()
	clusterplot.scatter(xplot, yplot, zplot, s=5, c='blue', linewidths=1)
	# plot the random negative samples; control the pct selected for plotting using frac
	dfrandsample = dfrand.sample(frac = 0.4)
	xplot = dfrandsample.a.tolist()
	yplot = dfrandsample.b.tolist()
	zplot = dfrandsample.h.tolist()
	clusterplot.scatter(xplot, yplot, zplot, s=2, c='grey', edgecolors='grey', linewidths=1)
	clusterplot.set_xlabel ('a',weight='bold')
	clusterplot.set_ylabel ('b',weight='bold')
	clusterplot.set_zlabel ('hypothenous',weight='bold')
	plt.show(block=False)

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

# ------------------ RUN PREDICTION IN MIXED TEST DATA -----------------------------------------
X = df_test[['a','b', 'h']].values 			#convert dataframe to np array for Keras
Y = df_test['label'].tolist() 				#convert pd series to np array for Keras
# do prediction
prediction = model.predict(X, batch_size = batch_size, verbose = 2)
round_prediction = [round(x[0]) for x in prediction]
#print(round_prediction[:10])
#print(prediction[:10])
#prediction_class = model.predict_classes(X, batch_size = batch_size, verbose = 2)
#print(prediction_class[:10])

# calculate confusion matrix 
print('------Prediction Performance of Mixed (Noise+Random) Data----------- \n')
#cm = confusion_matrix(Y, round_prediction)
#print('Confusion Matrix \n', cm)
tn, fp, fn, tp = confusion_matrix(Y, round_prediction).ravel()
print('TP', tp, 'FP', fp, 'TN', tn, 'FN', fn, '\n')
target_names = ['class 0', 'class 1']
print(classification_report(Y, round_prediction, target_names=target_names))

# ------------------ RUN PREDICTION IN NOISE & POSITIVE TEST DATA -----------------------------------------
DFNOISE = df_test[df_test['type'] != 'rand']  	# filter out all the rand entries in test dataset
X = DFNOISE[['a','b','h']].values  				#convert dataframe to np array for Keras
Y = DFNOISE['label'].tolist() 					#convert pd series to np array for Keras
# do prediction
prediction = model.predict(X, batch_size = batch_size, verbose = 2)
round_prediction = [round(x[0]) for x in prediction]

# calculate confusion matrix 
print('------Prediction Performance of Noise & Positive Data------------- \n')
tn, fp, fn, tp = confusion_matrix(Y, round_prediction).ravel()
print('TP', tp, 'FP', fp, 'TN', tn, 'FN', fn, '\n')
target_names = ['class 0', 'class 1']
print(classification_report(Y, round_prediction, target_names=target_names))

# ------------------ RUN PREDICTION IN RANDOM & POSITIVE TEST DATA -----------------------------------------
DFRAND = df_test[df_test['type'] != 'noise']  	# filter out all the noise entries in test dataset
X = DFRAND[['a','b','h']].values  				#convert dataframe to np array for Keras
Y = DFRAND['label'].tolist() 					#convert pd series to np array for Keras
# do prediction
prediction = model.predict(X, batch_size = batch_size, verbose = 2)
round_prediction = [round(x[0]) for x in prediction]

# calculate confusion matrix 
print('------Prediction Performance of Random & Positive Data------------- \n')
tn, fp, fn, tp = confusion_matrix(Y, round_prediction).ravel()
print('TP', tp, 'FP', fp, 'TN', tn, 'FN', fn, '\n')
target_names = ['class 0', 'class 1']
print(classification_report(Y, round_prediction, target_names=target_names))
