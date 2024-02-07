from numpy import random
import numpy as np 
x_train = random.random(size=30)
target_w = .2
target_b = 10
y_train = target_w*x_train +target_b

def linear_function(w,b,x):return np.add(np.multiply(w,x),b)
def cost_function(y_predicted,y_target):
    m=len(y_predicted)
    sum =np.power(np.subtract(y_predicted,y_target),2)
    return np.multiply(1/(2*m),sum)
def gradient_descent_algolrithm_w(y_pred,y_target,x_train):
    m = len(x_train)
    sum = np.sum(np.multiply(np.subtract(y_pred,y_target),x_train))
    return np.multiply(1/(2*m),sum)
def gradient_descent_algolrithm_b(y_pred,y_target):
    m = len(y_target)
    sum =np.sum(np.subtract(y_pred,y_target))
    return np.multiply(1/m,sum)
def main():
    w = 0
    b = 0
    epochs = 10000
    for epoch in range(epochs):
        learning_rate = .1
        y_pred = linear_function(w,b,x_train)
        temp_w= w - np.multiply(learning_rate,gradient_descent_algolrithm_w(y_pred=y_pred,y_target=y_train,x_train=x_train))
        temp_b= b - np.multiply(learning_rate,gradient_descent_algolrithm_b(y_pred=y_pred,y_target=y_train))
        w = temp_w
        b = temp_b
    print("-------Target---------")
    print(f"w:{target_w} b:{target_b}")
    print("------Predicted-------")
    print(f"w : {w} and b:{b}")
    pass
main()
