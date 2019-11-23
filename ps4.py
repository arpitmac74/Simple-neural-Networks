# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:12:59 2019

@author: Oshi
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

###Problem 1
###Provided function to create training data

def simplest_training_data(n):
  w = 3
  b = 2
  x = np.random.uniform(0,1,n)
  y = 3*x+b+0.3*np.random.normal(0,1,n)
  return (x,y)
def simplest_training(n, k, eta):
  mu, sigma = 0, 1 # mean and standard deviation
  w = np.random.normal(mu, sigma, 1)
  points=simplest_training_data(n)  
  theta=[w,0]
  for f in range(0,k):
      counter=0
      for i in range(0,n):
          theta[0]=theta[0]-eta*(2*(points[0][counter])*((theta[0]-3)*(points[0][counter])+theta[1]-2))
          theta[1]=theta[1]-eta*(2*((theta[0]-3)*(points[0][counter])+theta[1]-2))
          counter=counter+1
  return theta
def simplest_testing(theta,x):
    a=[]
    counter=0
    for i in range(0,9):
        y=theta[0]*x[counter]+theta[1]
        counter=counter+1
        a.append(y)
  #TODO: Your Code Here
    return a
#c=simplest_training(30,10000,0.02)
#y=simplest_testing(c,[0,1,2,3,4,5,6,7,8,9])
#print(y)





###Problem 2
###Provided function to create training data
def single_layer_training_data(trainset):
  n = 10
  if trainset == 1:
    # Linearly separable
    X = np.concatenate((np.random.normal((0,0),1,(n,2)), np.random.normal((10,10),1,(n,2))),axis=0)
    y = np.concatenate((np.ones(n), np.zeros(n)),axis=0)

  elif trainset == 2:
    # Not Linearly Separable
    X = np.concatenate((np.random.normal((0,0),1,(n,2)), np.random.normal((10,10),1,(n,2)), np.random.normal((10,0),1,(n,2)), np.random.normal((0,10),1,(n,2))),axis=0)
    y = np.concatenate((np.ones(2*n), np.zeros(2*n)), axis=0)

  else:
    print("function single_layer_training_data undefined for input", trainset)
    sys.exit()

  return (X,y)
k= single_layer_training_data(2)
#print(k[1])
#print(k[0][1][0])
#plt.plot(k[0],'+')
#plt.show()


def single_layer_training(k, eta, trainset):
  #TODO: Your Code Here
  mu, sigma = 0, 1 # mean and standard deviation
  w_1 = np.random.normal(mu, sigma, 1)
  w_2 = np.random.normal(mu, sigma, 1)
  points=single_layer_training_data(trainset)
  theta_new=[w_1,w_2,0]
  theta=[w_1,w_2,0]
    
  for c in range(0,k):
      counter=0
      for i in range(0,len(points[1])):
          theta_new[0]=theta[0]-eta*(((1/(1+(np.exp(-1*((theta[0]*(points[0][counter][0]))+(theta[1]*(points[0][counter][1]))+theta[2])))))-points[1][counter])*(points[0][counter][0]))
          theta_new[1]=theta[1]-eta*(((1/(1+(np.exp(-1*((theta[0]*(points[0][counter][0]))+(theta[1]*(points[0][counter][1]))+theta[2])))))-points[1][counter])*(points[0][counter][1]))
          theta_new[2]=theta[2]-eta*(((1/(1+(np.exp(-1*((theta[0]*(points[0][counter][0]))+(theta[1]*(points[0][counter][1]))+theta[2])))))-points[1][counter])*1)
          theta[0]=theta_new[0]
          theta[1]=theta_new[1]
          theta[2]=theta_new[2]
          counter=counter+1
  return theta,trainset

##
#def single_layer_testing(theta,trainset):
#    points=single_layer_training_data(trainset)
#    y=[]
#    counter=0
#    print(points[1])
#    for i in range(0,len(points[1])):
#        z=((theta[0]*(points[0][counter][0]))+(theta[1]*(points[0][counter][1]*theta[1])))+theta[2]
#        sigmoid=(1/(1+np.exp(-z)))
#        counter=counter+1
#        y.append(sigmoid)
#  #TODO: Your Code Here
#  
#    return y
  
def single_layer_testing(theta,trainset):
    points=np.array([[0, 0], [10, 0]]),np.array([0, 0])
    y=[]
    counter=0
    for i in range(0,len(points[1])):
        z=((theta[0]*(points[0][counter][0]))+((points[0][counter][1])*theta[1]))+theta[2]
        sigmoid=(1/(1+(np.exp(-z))))
        #print("z",z)
        counter=counter+1
        y.append(sigmoid)
  #TODO: Your Code Here
  
    return y
#theta,trainset=single_layer_training(10000, 0.02, 2)
##print(theta)
#y=single_layer_testing(theta,trainset)
#print(y)




###Problem 3
###Provided function to create training data
def pca_training_data(n, sigma):
  m = 1
  b = 1
  x1 = np.random.uniform(0,10,n)
  x2 = m*x1+b
  X = np.array([x1,x2]).T
  X += np.random.normal(0,sigma,X.shape)
  return X



def pca_training(k, eta):
  #TODO: Your Code Here
  c=pca_training_data(10, 0.1)
  x=np.round(c,3)
  plt.plot(x)
  plt.show()
  mu, sigma = 0, 1 # mean and standard deviation
  w11=0.03
  w12=0.02
  b11=0
  w21=0.01
  w22=0.02
  b21=0
  b22=0
  
  th=np.round([w11,w12,b11,w21,w22,b21,b22],3)
  th_new=np.round([w11,w12,b11,w21,w22,b21,b22],3)
  for c in range(0,k):
      counter=0
      for i in range(0,len(x)):
          h=th[0]*x[counter][0]+th[1]*x[counter][1]+th[2]
          z1 = (th[3]*h)+th[5]
          z2 = (th[4]*h)+th[6]
          th_new[0]=th[0]-eta*(2*x[counter][0]*((z1-x[counter][0])*w21+(z2-x[counter][1])*w22))
          th_new[1]=th[1]-eta*(2*x[counter][1]*((z1-x[counter][0])*w21+(z2-x[counter][1])*w22))
          th_new[2]=th[2]-eta*(2*((z1-x[counter][0])*w21+(z2-x[counter][1])*w22))
          th_new[3]=th[3]-eta*(2*h*(z1-x[counter][0]))
          th_new[4]=th[4]-eta*(2*h*(z2-x[counter][1]))
          th_new[5]=th[5]-eta*(2*(z1-x[counter][0]))
          th_new[6]=th[6]-eta*(2*(z2-x[counter][1]))
          th[0]=th_new[0]
          th[1]=th_new[1]
          th[2]=th_new[2]
          th[3]=th_new[3]
          th[4]=th_new[4]
          th[5]=th_new[5]
          th[6]=th_new[6]
          counter+=1
          
  return th,x




def pca_test(th, X):
  #TODO: Your Code Here
  Z=[]
  counter=0
  for i in range(0,len(X)):
        h=th[0]*X[counter][0]+th[1]*X[counter][1]+th[2]
        z1 = (th[3]*h)+th[5]
        z2 = (th[4]*h)+th[6]
        z=[z1,z2]
        Z.append(z)
        counter=counter+1
  return Z
o,x=pca_training(2000, 0.03)
test=pca_test(o, [[1,2], [4,5], [10, 3]])
print(test)



###Problem 4: Challenge Problem
def nn_training(k, eta, trainset, H):
  #TODO: Your Code Here
  return theta

def nn_testing(theta, X):
  #TODO: Your Code Here
  return y

