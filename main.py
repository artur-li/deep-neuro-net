import tensorflow as tf    

# DATA
# upload dataset
mnist = tf.keras.datasets.mnist 
# split dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# edit dataset values to be between 0-1 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# MODEL ARCHITECTURE
# model type
model = tf.keras.models.Sequential()
# input layer
model.add(tf.keras.layers.Flatten())
# hidden layer 1
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# hidden layer 2
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# TRAINING
# configures model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# trains the model
model.fit(x_train, y_train, epochs=3)

# EVALUATION
# evaluate the model... returns loss and accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc )
