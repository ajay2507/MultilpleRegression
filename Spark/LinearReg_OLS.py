# LinearReg_OLS.py  
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the 
# closed form expression for the ordinary least squares estimate of beta.
# Takes the yx file as input, where on each line y is the first element # and the remaining elements constitute the x. 
# Author : Ajaykumar Prathap
# Email : aprathap@uncc.edu

import sys 
import numpy as np
from pyspark import SparkContext



   # Method to find X*X.T to calculate A
def calcA(l):
   # Make the initial column 1 to manage beta co-efficient
   l[0] = 1.0
   # create an array from the given input
   temp_array = np.array(l).astype('float')
   # take transponse and convert that as matrix
   X = np.asmatrix(temp_array).T
   # Matrix Multiplication of X & X.T 
   final_X = np.dot(X,X.T)
   # Return final value
   return final_X

 #Method to find X*Y to calculate B
def calcB(l):
   # consider first column as Y
   Y = float(l[0])
   # Make the initial column 1 to manage beta co-efficient
   l[0] = 1.0
   # create an array from the given input
   temp_array = np.array(l).astype('float')
   # take transponse and convert that as matrix
   X = np.asmatrix(temp_array).T
   # Matrix Multiplication of X & Y
   final_X = np.multiply(X,Y)
   # Return final value
   return final_X

if __name__ == "__main__":   
  if len(sys.argv) !=2:     
     print >> sys.stderr, "Usage: linreg <datafile>"     
     exit(-1)
 
  sc = SparkContext(appName="LinearRegression")
  # Input yx file has y_i as the first element of each line   
  # and the remaining elements constitute x_i 
  # Reading the input file
  yxinputFile = sc.textFile(sys.argv[1])
  # Spliting the values by comma
  yxlines = yxinputFile.map(lambda line: line.split(','))  
  
  # Construct A Mapper and keep key as KeyA
  A_map = yxlines.map(lambda l: ("KeyA",calcA(l)))
  # Add all the values for the key KeyA and collect 
  A_reduce = A_map.reduceByKey(lambda x,y: np.add(x,y)).map(lambda l: l[1]).collect()[0]
  # construst A matrix 
  A = np.asmatrix(A_reduce)

  B_map = yxlines.map(lambda l: ("KeyB",calcB(l)))
  B_reduce = B_map.reduceByKey(lambda x,y: np.add(x,y)).map(lambda l: l[1]).collect()[0]
  # construst B matrix 
  B = np.asmatrix(B_reduce)
  
  final_matrix = np.dot(np.linalg.inv(A),B)
  beta = np.array(final_matrix).tolist()
  
  # print the linear regression coefficients
  print "beta: "   
  for coeff in beta:       
  	print coeff
  sc.stop()
