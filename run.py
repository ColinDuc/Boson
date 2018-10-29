import numpy as np
from implementation import *
from utilities import *


print('Loading the train and test data.')
y_train,x_train_raw,ids_train=load_csv_data("train.csv") 
y_test,x_test_raw,ids_test=load_csv_data("test.csv")

print('Data cleaning and processing')
x_train=data_final(x_train_raw)
x_test=data_final(x_test_raw)
x_train_train, x_train_test, y_train_train, y_train_test = split_data(x_train,y_train,0.8) #Creating a training and a test set to chose the best weights by partitioning (80%/20%) the training set

weight,lambda_=select_para_ridge(y_train_train,x_train_train,y_train_test,x_train_test) #selecting the best weights by using ridge regression

print('Performing prediction based on the model')
final_prediction=definitive_res(_test.dot(weight))

print('Creating the final file with the predictions')
create_csv_submission(ids_test,pred,"Final_Submission")