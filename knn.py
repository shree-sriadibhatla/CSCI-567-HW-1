import numpy as np
import json
import collections
import matplotlib.pyplot as plt


def data_processing(data):
	train_set, valid_set, test_set = data['train_data'], data['val_data'], data['test_data']
	Xtrain = train_set["features"]
	ytrain = train_set["labels"]
	Xval = valid_set["features"]
	yval = valid_set["labels"]
	Xtest = test_set["features"]
	ytest = test_set["labels"]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False):
	train_set, valid_set, test_set = data['train_data'], data['val_data'], data['test_data']
	Xtrain = train_set["features"]
	ytrain = train_set["labels"]
	Xval = valid_set["features"]
	yval = valid_set["labels"]
	Xtest = test_set["features"]
	ytest = test_set["labels"]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	# We load data from json here and turn the data into numpy array
	# You can further perform data transformation on Xtrain, Xval, Xtest

	# Min-Max scaling
	if do_minmax_scaling:
		# pass
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		e = 1e-8
		Xtrain_min = Xtrain.min(axis=0)
		Xtrain_max = Xtrain.max(axis=0)
		Xtrain = (Xtrain - Xtrain_min)/(Xtrain_max-Xtrain_min+e)
		Xval = (Xval - Xtrain_min)/(Xtrain_max-Xtrain_min+e)
		Xtest = (Xtest - Xtrain_min)/(Xtrain_max-Xtrain_min+e)



	# Normalization
	def normalization(x):
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		e = 1e-8
		x_l2 = np.sqrt(np.sum(x**2,axis=1))+e
		x = x[:,:]/x_l2[:,None]
		return x
	
	if do_normalization:
		Xtrain = normalization(Xtrain)
		Xval = normalization(Xval)
		Xtest = normalization(Xtest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def compute_l2_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	X_l2_square = np.sum(X**2, axis=1).reshape(-1,1)
	Xtrains_l2_square = np.sum(Xtrain**2, axis=1).reshape(1,-1)
	dists = np.sqrt(X_l2_square+Xtrains_l2_square-2*(X@Xtrain.T))
	return dists


def compute_cosine_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Cosine distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	e=1e-8
	Xtrain_dot_X = X@Xtrain.T
	Xtrain_l2 = np.sqrt(np.sum(Xtrain**2,axis=1)).reshape(-1,1)+e
	X_l2 = np.sqrt(np.sum(X**2,axis=1)).reshape(-1,1)+e
	dists=1-(Xtrain_dot_X/(X_l2@Xtrain_l2.T))
	return dists


def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	ypred = np.zeros(dists.shape[0], dtype=ytrain.dtype)
	for i in range(dists.shape[0]):
		x = dists[i]
		top_idx = np.argpartition(x, k)[:k]
		pred = 0
		for idx in top_idx:
			pred += ytrain[idx]
		ypred[i] = np.where(pred<=k/2,0,1)
	return ypred


def compute_error_rate(y, ypred):
	"""
	Compute the error rate of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- err: The error rate of prediction (scalar).
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	result = np.zeros(y.shape, dtype=y.dtype)
	result = np.where(y!=ypred,1,0)
	err = np.sum(result)/y.shape[0]
	return err


def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation error rate.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the lowest error rate.
	- validation_error: A list of error rate of different ks in K.
	- best_err: The lowest error rate we get from all ks in K.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	validation_error = []
	best_err = 100
	best_k = 0
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		err = compute_error_rate(yval, ypred)
		validation_error.append(err)
		if best_err > err:
			best_err = err
			best_k = k
	return best_k, validation_error, best_err


def main():
	input_file = 'breast_cancer_dataset.json'
	output_file = 'knn_output.txt'

	#==================Problem Set 1.1=======================

	with open(input_file) as json_data:
		data = json.load(json_data)

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.1")
	print()

	#==================Problem Set 1.2=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=False, do_normalization=True)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using normalization")
	print()

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using minmax_scaling")
	print()
	
	#==================Problem Set 1.3=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	dists = compute_cosine_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.3, which use cosine distance")
	print()

	#==================Problem Set 1.4=======================
	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	#======performance of different k in training set=====
	K = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	dists = compute_l2_distances(Xtrain, Xtrain)
	train_error = []
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		err = compute_error_rate(ytrain, ypred)
		train_error.append(err)
	#==========select the best k by using validation set==============
	dists = compute_l2_distances(Xtrain, Xval)
	best_k, validation_error, best_err = find_best_k(K, ytrain, dists, yval)
			
	plt.figure()
	plt.title('training and validation error vs k')
	plt.xlabel('k')
	plt.ylabel('error rate')
	plt.plot(K,train_error,marker='o',label='training error')
	plt.plot(K,validation_error,marker='o',label='validation error')
	plt.legend() 
	plt.xticks(K)
	plt.grid(True)
	plt.show()

	#===============test the performance with your best k=============
	dists = compute_l2_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_err = compute_error_rate(ytest, ypred)
	print("In Problem Set 1.4, we use the best k = ", best_k, "with the best validation error rate", best_err)
	print("Using the best k, the final test error rate is", test_err)
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_error[i])+'\n')
	f.write('%s %.3f' % ('test', test_err))
	f.close()

if __name__ == "__main__":
	main()
