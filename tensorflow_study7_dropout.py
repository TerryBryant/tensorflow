import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

def add_layer(inputs, in_size, out_size, name=None, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')   #初始变量用随机值比用0好得多
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob) #dropout
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    # tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

#define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32) #用于dropout
xs = tf.placeholder(tf.float32, [None, 64]) #8*8
ys = tf.placeholder(tf.float32, [None, 10])


# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax) #softmax一般用于分类

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=1))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

#summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# important step
sess.run(tf.global_variables_initializer())  #非常重要的一步，激活init

for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob:0.6}) #每次drop掉40%
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob:1}) #记录时不能drop任何东西
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob:1})
        train_writer.add_summary(test_result, i)
        test_writer.add_summary(test_result, i)
