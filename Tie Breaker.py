
# coding: utf-8

# In[27]:


import pandas as pd
data_xls = pd.read_excel('main dataset.xlsx', 'Sheet1', index_col=None)
data_xls.to_csv('md.csv', encoding='utf-8')


# In[14]:


import numpy as np
df1 = pd.read_csv('player1.csv', index_col = False)
df2 = pd.read_csv('player2.csv', index_col = False)


# In[12]:


df1


# In[20]:


del df1["player1 m"]


# In[21]:


df1


# In[23]:


df2 = df2.rename(columns={'player2 height': 'player1 cm'})


# In[24]:


df2


# In[97]:


df1


# In[68]:


df1.to_csv('final.csv', index=False)


# In[123]:


import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[124]:


data = np.genfromtxt('final1.csv', delimiter=',')

#Get the 2 features (hours slept & hours studied)
X = data[:, 1:11]
# Get the result (0 suspended - 1 approved)
Y = data[:,0]
#Split the data in train & test
Y_reshape = data[:,0].reshape(data[:,0].shape[0], 1)
x_train, x_test, y_train, y_test = train_test_split(data[:, 1:11], Y_reshape)

print ("x_train shape: " + str(x_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("x_test shape: " + str(x_test.shape))
print ("y_test shape: " + str(y_test.shape))

num_features = x_train.shape[1]


# In[125]:


data.shape


# In[126]:


num_features


# In[127]:


X.shape


# In[128]:


Y


# In[130]:


learning_rate = 0.005
training_epochs = 2000
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, num_features], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

# Initialize our weigts & bias
W = tf.get_variable("W", [num_features, 1], initializer = tf.contrib.layers.xavier_initializer())
b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())

#Z = tf.add(tf.matmul(X, W), b)
#prediction = tf.nn.sigmoid(Z)

prediction = tf.nn.softmax(tf.matmul(X, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(prediction), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))


#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                "W=", sess.run(W), "b=", sess.run(b))
        cost_history = np.append(cost_history, c)
        
        
    # getting predictions
    correct_prediction = tf.to_float(tf.greater(prediction, 0.5))

   #acc
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, correct_prediction)))

    print ("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
    print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

