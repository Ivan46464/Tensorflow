import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential


# Define the metrics
accuracy = tf.keras.metrics.Accuracy()
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
auc = tf.keras.metrics.AUC()
df = pd.read_csv("diabetes.csv")
print(df.info())
features = df.iloc[:, 0:8]
label = df["Outcome"]

print(features.columns)
features_train, features_test, labels_train, labels_test = train_test_split(features, label, test_size=0.33, random_state=42) #split the data into training and test data
ct = ColumnTransformer([('standardize', StandardScaler(), features.columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)
model = Sequential(name="my_first_model")
input = tf.keras.layers.InputLayer(input_shape=(features.shape[1],))
model.add(input)
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1))
print(model.summary())
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(features_train , labels_train , epochs=100, batch_size=1)
val_loss, val_accuracy = model.evaluate(features_test, labels_test, verbose=0)
print("The loss is: " + str(val_loss))
print("The val accuracy is: " + str(val_accuracy))

