import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer() #如果定义了变量，必须用这个函数

with tf.Session() as sess:
    sess.run(init) #必须初始化
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) #直接print(state)没用，必须run