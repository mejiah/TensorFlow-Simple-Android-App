import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph

NUM_FEATURES = 2
NUM_ITER = 2000
learning_rate = 0.01

x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) # 4x2, input
print(x)
y = np.array([0, 0, 1, 0], np.float32) # 4, correct output, AND operation
#y = np.array([0, 1, 1, 1], np.float32) # OR operation
y = np.reshape(y, [4,1]) # convert to 4x1

X = tf.placeholder(tf.float32, shape=[4, 2])
Y = tf.placeholder(tf.float32, shape=[4, 1])

userInputA = tf.placeholder(tf.float32, name="modelInputA")
userInputB = tf.placeholder(tf.float32, name="modelInputB")

W = tf.Variable(tf.zeros([NUM_FEATURES, 1]), tf.float32)
B = tf.Variable(tf.zeros([1, 1]), tf.float32)

yHat = tf.sigmoid(tf.add(tf.matmul(X, W), B))  # 4x1
err = Y - yHat
deltaW = tf.matmul(tf.transpose(X), err)  # have to be 2x1
deltaB = tf.reduce_sum(err, 0)  # 4, have to 1x1. sum all the biases? yes
W_ = W + learning_rate * deltaW
B_ = B + learning_rate * deltaB

step = tf.group(W.assign(W_), B.assign(B_))  # to update the values of weights and biases.

prediction = tf.add(tf.add(tf.multiply(userInputA, W[0]), tf.multiply(userInputB, W[1])), B, name="modelOutput")  # 4x1

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    for k in range(NUM_ITER):
        sess.run([step], feed_dict={X: x, Y: y})

    #print(sess.run(prediction, feed_dict={userInputA: [0], userInputB: [1]})[0][0])
    #print(sess.run(W[0]) * 1 + sess.run(W[1]) * 1 + sess.run(B[0]))

    out = saver.save(sess, 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/results.ckpt', global_step=1)
    tf.train.write_graph(sess.graph_def, 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/', 'results.pbtxt')
    tf.train.write_graph(sess.graph_def, 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/','results.pb', as_text=False)

# Freeze the graph

# graph definition saved above
input_graph = 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/results.pb'
# any other saver to use other than default
input_saver = ""
# earlier definition file format text or binary
input_binary = True
# checkpoint file to merge with graph definition
input_checkpoint = 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/results.ckpt-1'
# output nodes inn our model
output_node_names = 'modelOutput'
restore_op_name = 'save/restore_all'
filename_tensor_name = 'save/Const:0'
# output path
output_graph = 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/'+'frozen_'+'results'+'.pb'
# default True
clear_devices = True
initializer_nodes = ""
variable_names_blacklist = ""

freeze_graph.freeze_graph(
    input_graph,
    input_saver,
    input_binary,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist
)


"""
w = tf.Variable(10, name="test")
ten = tf.constant(10)
new_var = tf.multiply(w, ten)
update = tf.assign(w, new_var)
new_var2 = tf.add(ten, ten)

pla1 = tf.placeholder(tf.float32)
pla2 = tf.placeholder(tf.float32)
result = tf.multiply(pla1, pla2)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # Run the Op that initializes global variables.
    sess.run(init_op)
    temp = sess.run([new_var, new_var2])
    print(sess.run(result, feed_dict={pla1:[100], pla2:[100]})[0])
    # ...you can now run any Op that uses variable values...
"""

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.tools import freeze_graph


NUM_FEATURES = 2
NUM_ITER = 2000
learning_rate = 0.01

x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)  # 4x2, input
y = np.array([0, 0, 1, 0], np.float32)  # 4, correct output, AND operation
# y = np.array([0, 1, 1, 1], np.float32) # OR operation
y = np.reshape(y, [4, 1])  # convert to 4x1

X = tf.placeholder(tf.float32, shape=[4, 2])
Y = tf.placeholder(tf.float32, shape=[4, 1])

W = tf.Variable(tf.zeros([NUM_FEATURES, 1]), tf.float32)
B = tf.Variable(tf.zeros([1, 1]), tf.float32)

yHat = tf.sigmoid(tf.add(tf.matmul(X, W), B), name="modelOutput")  # 4x1
err = Y - yHat
deltaW = tf.matmul(tf.transpose(X), err)  # have to be 2x1
deltaB = tf.reduce_sum(err, 0)  # 4, have to 1x1. sum all the biases? yes
W_ = W + learning_rate * deltaW
B_ = B + learning_rate * deltaB

step = tf.group(W.assign(W_), B.assign(B_))  # to update the values of weights and biases.

A2 = tf.placeholder(tf.float32, shape=[1], name='modelInputA') # input a
B2 = tf.placeholder(tf.float32, shape=[1], name='modelInputB') # input b

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

for k in range(NUM_ITER):
    sess.run([step], feed_dict={X: x, Y: y})

out = saver.save(sess, 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/results.ckpt', global_step=1)
tf.train.write_graph(sess.graph_def, 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/', 'results2.pbtxt')
tf.train.write_graph(sess.graph_def, 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/','results3.pb', as_text=False)


# Freeze the graph

# graph definition saved above
input_graph = 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/results3.pb'
# any other saver to use other than default
input_saver = ""
# earlier definition file format text or binary
input_binary = True
# checkpoint file to merge with graph definition
input_checkpoint = 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/results.ckpt-1'
# output nodes inn our model
output_node_names = 'modelOutput'
restore_op_name = 'save/restore_all'
filename_tensor_name = 'save/Const:0'
# output path
output_graph = 'C:/Users/hmeji/Desktop/androidDev/tensorflow_demo/DeepLearningOnAndroid/DeepLearningOnAndroid/savedFiles/'+'frozen_'+'results3'+'.pb'
# default True
clear_devices = True
initializer_nodes = ""
variable_names_blacklist = ""

freeze_graph.freeze_graph(
    input_graph,
    input_saver,
    input_binary,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist
)
"""

"""
# Now plot the fitted line. We need only two points to plot the line
plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1] + 0.2)])
plot_y = - 1 / W[1] * (W[0] * plot_x + b)
plot_y = np.reshape(plot_y, [2, -1])
plot_y = np.squeeze(plot_y)

print('W: ' + str(W))
print('b: ' + str(b))
print('plot_y: ' + str(plot_y))

plt.scatter(x[:, 0], x[:, 1], c=y, s=100, cmap='viridis')
plt.plot(plot_x, plot_y, color='k', linewidth=2)
plt.xlim([-0.2, 1.2]);
plt.ylim([-0.2, 1.25]);
plt.show()
"""

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print('TensorFlow version: ' + tf.__version__)


from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("../MNIST_data", one_hot=True, validation_size=0)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

print ('We have '+str(x_train.shape[0])+' training examples in dataset')
print ('We have '+str(x_train.shape[1])+' feature points(basically pixels) in each input example')

TUTORIAL_NAME = 'Tutorial2'
MODEL_NAME = 'mnistTFonAndroid'
SAVED_MODEL_PATH = '../' + TUTORIAL_NAME+'_Saved_model/'


LEARNING_RATE = 0.1
TRAIN_STEPS = 2000

# Our single node softmax model
X = tf.placeholder(tf.float32, shape=[None, 784], name='modelInput')
Y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]), name='modelWeights')
b = tf.Variable(tf.zeros([10]), name='modelBias')
Y = tf.nn.softmax(tf.matmul(X,W) + b, name='modelOutput')

# Training and performance matrices
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Lets initialize the vairables and train our model
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
saver = tf.train.Saver()


for i in range(TRAIN_STEPS+1):
    sess.run(training, feed_dict={X: x_train, Y_: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) +
              '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={X: x_test, Y_: y_test})) +
              '  Loss = ' + str(sess.run(cross_entropy, {X: x_train, Y_: y_train}))
             )
    if i%500 == 0:
        out = saver.save(sess, SAVED_MODEL_PATH + MODEL_NAME + '.ckpt', global_step=i)

"""