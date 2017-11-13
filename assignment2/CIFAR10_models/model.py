import tensorflow as tf
'''
In this file we will write an implementation of a model that is quite
similar to the model that we implemented in the tensorflow notebook,we will
test the influence of the filter size(which was 7 in the model that we implemented in the notebook)
on the test scores and efficiency of the model.
The model depth will stay similar:
1.Conv layer:NUM_FILTERS filters of size FxF with stride:S
2.Relu activation.
3.Spatial batchnorm.
4.Max pooling of size max_pool_size with stride:max_pool_s.
5.Affine layer with HIDDEN output units.
6.Relu activation.
7.Affine layer to convert HIDDEN input units to 10 output units.
Note:We will also change the optimizer from RMSProp to Adam.(Adam uses a combination of RMSProp with momentum)
For now the padding will be set to VALID.
'''
#Initialize number of filters.
NUM_FILTERS = 64
#Size of the filters.
F = 5
#stride to use.
S = 2
#Size of max_pool.
max_pool_size = 2
#Stride for max pooling.
max_pool_s = 2
HIDDEN = 512
def my_model(X,y,is_training):
	#Setup variables.
    #Define convolutional layer.
    Wconv1 = tf.get_variable("Wconv1",shape = [F,F,3,NUM_FILTERS])
    bconv1 = tf.get_variable("bconv1",shape = [NUM_FILTERS])
    #Compute output size after Conv layer and max_pool layer.(Width and height are the same)
    after_conv = (32 - F) / S + 1
    after_pool = int((after_conv - max_pool_size) / max_pool_s + 1) 
    #Fully connected layers.
    W1_size = (after_pool ** 2) * NUM_FILTERS
    W1 = tf.get_variable("W1",shape = [W1_size,HIDDEN])
    b1 = tf.get_variable("b1",shape = [HIDDEN])
    W2 = tf.get_variable("W2",shape = [HIDDEN,10])
    b2 = tf.get_variable("b2",shape = [10])
    
    #Define the graph.
    a1 = tf.nn.conv2d(X,Wconv1,strides = [1,S,S,1],padding = 'VALID') + bconv1
    h1 = tf.nn.relu(a1)
    #Run spatial batch normalization.
    h1_flat = tf.reshape(h1,[-1,NUM_FILTERS])
    #Compute moments.
    mean,variance = tf.nn.moments(h1_flat,axes = [0])
    gamma = tf.get_variable("gamma",shape = [NUM_FILTERS])
    beta = tf.get_variable("beta",shape = [NUM_FILTERS])
    #Batch normalize.
    h2_flat = tf.nn.batch_normalization(h1_flat,mean,variance,beta,gamma,1e-8)
    h2 = tf.reshape(h2_flat,tf.shape(h1))
    #2x2 max pooling
    h3 = tf.nn.max_pool(h2,ksize = [1,max_pool_size,max_pool_size,1],\
    				strides = [1,max_pool_s,max_pool_s,1],padding="VALID")
    #Affine layer.
    h3_flat = tf.reshape(h3,[-1,W1_size])
    h4 = tf.matmul(h3_flat,W1) + b1
    a2 = tf.nn.relu(h4)
    y_out = tf.matmul(a2,W2) + b2
    return y_out

tf.reset_default_graph()

#Set placeholders for input data and labels.
X = tf.placeholder(tf.float32,[None,32,32,3])
y = tf.placeholder(tf.float32,[None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
	