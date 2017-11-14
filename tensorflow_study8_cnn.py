import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variale(shape):
    initial = tf.constant(0.1, shape=shape) #bias一般用正值
    return tf.Variable(initial)

def conv2d(x, W):
    # x为输入值，W为权重
    # stride为步长，第一四个值为1，第二三个值为x、y方向步长
    # padding选same表示在图片周围补若干圈0，这样抽取后的图像与原图像尺寸完全一致（为什么是若干圈，因为stride不同）
    # padding选valid表示在图片直接抽取，抽取后的图像比原图像小一圈
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#define placeholder
xs = tf.placeholder(tf.float32, [None, 784]) #28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1]) #-1表示所有图片的维度，28*28，1表示当前图片为黑白，channel
# print(x_image.shape) #[n_samples,28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) #5*5是patch尺寸，1表示输入图片的厚度为1，32表示输出的厚度为32
b_conv1 = bias_variale([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #输出的尺寸为28*28*32
h_pool1 = max_pool_2x2(h_conv1)#输出的尺寸为14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) #5*5是patch尺寸，输入图片的厚度为32，输出的厚度为64
b_conv2 = bias_variale([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #输出的尺寸为14*14*64
h_pool2 = max_pool_2x2(h_conv2)#输出的尺寸为7*7*64

## func1 layer ##
W_func1 = weight_variable([7*7*64, 1024])
b_func1 = bias_variale([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_func1) + b_func1)
h_func1_drop = tf.nn.dropout(h_func1, keep_prob=keep_prob) #训练数据较小时，为防止过拟合，选用dropout

## output layer ##
W_func2 = weight_variable([1024, 10])
b_func2 = bias_variale([10])

prediction = tf.nn.softmax(tf.matmul(h_func1_drop, W_func2) + b_func2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #系统太庞大，用admoptimize较好

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())  #非常重要的一步，激活init

# compute accurary
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        train_accurary = sess.run(accurary, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0})
        print("step %d, training accurary %.5f" % (i, train_accurary))
        # print(compute_accuracy(mnist.test.images, mnist.test.labels))
