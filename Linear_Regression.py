#Tensorflow Project Linear Regression on Titanic Data Set
#Using Google's Data set 
#Accuracy caps at approx 78% using this method with increased amount of epochs


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf




dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # Training Data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # Testing Data
#print(dftrain.head()) #Shows the first part of the data frame
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#print(dftrain.describe()) #Gives Statistics on the data frame
#print(dftrain.shape)

##############################################################################################################
#Some graphical interpretations of the data

dftrain.age.hist(bins=20)
#plt.show()
dftrain.sex.value_counts().plot(kind='barh')
#plt.show()
dftrain['class'].value_counts().plot(kind='barh')
#plt.show()
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#plt.show()

#############################################################################################################

CATCOL = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMCOL = ['age', 'fare']

feature_columns = []
for feature_name in CATCOL:
  vocabulary = dftrain[feature_name].unique()  #Gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMCOL:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#print(feature_columns) #(DEBUG)
#print(dftrain['sex'].unique())


def make_input_fn(data_df, label_df, num_epochs=50, shuffle=True, batch_size=128):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  #Create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  #Randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # Return a batch of the dataset
  return input_function  

train_input_fn = make_input_fn(dftrain, y_train)  
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
#Linear estimator based off of feature columns

linear_est.train(train_input_fn)  #Train model
result = linear_est.evaluate(eval_input_fn)  #Get model metrics/stats by testing on testing data

clear_output() 
print("=================================================================================")
print("Accuracy: ", result['accuracy'])
print("=================================================================================")

result = list(linear_est.predict(eval_input_fn))


print("=========================================")
for i in range(len(result)):
  print((dfeval.loc[i]))
  print("Did the passenger survive?: ",y_eval.loc[i])
  print("Predicted chance of survival: ",result[i]["probabilities"][1])
  print("=========================================")





