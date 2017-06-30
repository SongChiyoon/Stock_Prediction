import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data, wb
import datetime
sequence = 7
inputD = 5
outD = 1
#CODE BY SONG
def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def dataConvert(data):
    data = data[::-1]
    data = MinMaxScaler(data)
    x = data
    y = data[:, [-1]]
    x_data = []
    y_data = []
    for i in range(0, len(x) - sequence):
        _x = x[i:i + sequence]
        _y = y[i + sequence]

        x_data.append(_x)
        y_data.append(_y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


#collecting data set
start = datetime.datetime(2010,1,1)
end = datetime.datetime.now()
stocks =["KRX:005380", "NASDAQ:AAPL", "NASDAQ:GOOGL", "KRX:005930"]

print(len)
df = []
for i,stock in enumerate(stocks):
    df.append( data.DataReader(
        stock,"google",start,end
    ))
df[0].as_matrix()
data = np.loadtxt('TSLA.csv', delimiter=',')
x_data, y_data = dataConvert(data)



print(y_data.shape)
train_size = int(len(y_data) * 0.7)

x_train = np.array(x_data[:train_size])
x_test = np.array(x_data[train_size:])

y_train = np.array(y_data[:train_size])
y_test = np.array(y_data[train_size:])

print('tx:', x_train.shape,'tex:',x_test.shape,'ty:',y_train.shape,'tey:',y_test.shape)

X = tf.placeholder(tf.float32, [None, None, inputD])
Y = tf.placeholder(tf.float32, [None, outD])

cell = rnn.BasicLSTMCell(num_units=10, state_is_tuple=True, activation=tf.tanh)

outputs, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], outD)

loss = tf.reduce_sum(tf.square(y_pred - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)



targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

otherData = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
otherX, otherY = dataConvert(otherData)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1001):
        _, step_loss = sess.run([train, loss], feed_dict={X: x_train, Y: y_train})
        if(i % 100 == 0):
          print(i, step_loss)

    result = sess.run(y_pred, feed_dict={X:otherX})
    rmse = sess.run(rmse, feed_dict={targets:result, predictions:otherY})


print("rmse:",rmse)

plt.plot(otherY, 'red')
plt.plot(result)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()
