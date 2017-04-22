# GradientDescent.py  
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by Gradient Descent Approach
# considering alpha and Beta value dynamically so that beta converges 
# at one point
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

	yxfirstline = yxlines.first()

	# Find length of first line
	yxlength = len(yxfirstline)

	# Finding the values of A and B inorder to substitute the values
	# in Beta formula (Gradient Descent)



	


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



  	oldBeta = np.zeros(yxlength, dtype=float)
  	# set flag to True initially
  	flag = True
  	# set count to 0
  	count = 0
  	# set value of Alpha
  	alpha = 0.001


  	while flag:
    	 count = count+1
    	 #compute new beta value
    	 value = np.subtract(np.multiply(alpha,B),reduce(np.dot, [alpha,A,oldBeta]))
    	 newBeta = np.add(oldBeta,value)
         # if number of iterations reaches 500 then make alpha divide by 10
    	 if(count == 500) and flag:
      		count = 0
      		alpha = alpha/10
      	 # check the both the beta are equal or not
    	 if ((oldBeta == newBeta).all()):
        	flag = False
        	break
    	 else:
      		oldBeta = newBeta

        # Print the Values of Beta
  	print "beta: "
        beta = np.array(newBeta).tolist()
        #print newBeta  
	for coeff in beta:       
	 	print coeff[0]
	sc.stop()
