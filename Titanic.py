
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd        
import tensorflow as tf
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# In[4]:


df = pd.read_csv(r'C:\Users\jfh470\Desktop\titanic\train.csv')
# randomizes data
df = df.sample(frac=1).reset_index(drop=True)

# fills empty spaces
df['Embarked']=df['Embarked'].fillna('S') 
df['Age']=df['Age'].fillna(df['Age'].dropna().median())
df['Fare'] = df['Fare'].fillna(df['Fare'].dropna().median)

# changes categorical data to numerical
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# drops unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# splits data into test and train
df_train, df_test = np.split(df, [int(.7*len(df))])

df_test = df_test.reset_index(drop=True)

print(df_train.info())
df_test.info()


# In[5]:


X_train = df_train.drop(['Survived'], axis=1)
y_train = df_train['Survived']


# In[6]:


X_train.head(5)


# In[7]:


X_train = X_train.values
X_test = df_test.drop(['Survived'], axis=1).values
y_test = df_test['Survived'].values

print(X_test)


# Gradient Boosting

# In[6]:


from sklearn.ensemble import GradientBoostingClassifier
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
n_estimators = [50, 100, 150, 200, 250]
for learning_rate in learning_rates:
    for n_estimator in n_estimators:
        gb = GradientBoostingClassifier(n_estimators=n_estimator, learning_rate=learning_rate, max_features=7, max_depth=7)
        gb.fit(X_train, y_train)
        #     gb_predict = pd.DataFrame(gb.predict(X_test), columns=['Gradient Boost Predictions'])
        print(n_estimator, " estimators, learning rate of ", learning_rate, ", accuracy = ", gb.score(X_test, y_test))


# Random Forest

# In[8]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=871, max_features=5)
random_forest.fit(X_train, y_train)
rand_for_predict = random_forest.predict(X_test)
print(rand_for_predict)
rand_for_predict = pd.DataFrame(rand_for_predict, columns=['Random Forest Predictions'])
print(random_forest.score(X_test, y_test))



# XGBoost

# In[9]:


from xgboost import XGBClassifier
xgb_model = xgboost.XGBClassifier(n_estimators=500, learning_rate=0.001, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
xgb_predict = pd.DataFrame(xgb_model.predict(X_test), columns=['XGBoost Predictions'])

print(xgb_predict)
xgb_model.score(X_test, y_test)


# Neural Network

# In[10]:


def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:], y_data[idxs]


# In[11]:


epochs = 250
batch_size = 100
X_test = tf.Variable(X_test)


# In[12]:


w1 = tf.Variable(tf.random.normal([7, 128], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random.normal([128]), name='b1')

w2 = tf.Variable(tf.random.normal([128, 10]), name='w2')
b2 = tf.Variable(tf.random.normal([10]), name='b2')


# In[13]:


def nn_model(x_input, w1, b1, w2, b2):
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), w1), b1)
#     print(x)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, w2), b2)
    return logits


# In[14]:


def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy


# In[15]:


optimizer = tf.keras.optimizers.Adam()


# In[16]:


total_batch = int(len(y_train)/batch_size)
accuracy = 0
previous = -2
for epoch in range(epochs):
    if accuracy > previous + 1:
        avg_loss = 0
        nn_predict = list()
        for i in range(total_batch):
            batch_x, batch_y = get_batch(X_train, y_train, batch_size=batch_size)
            batch_x = tf.Variable(batch_x)
            batch_y = tf.Variable(batch_y)
            batch_y = tf.one_hot(batch_y, 10)
            with tf.GradientTape() as tape:
                logits = nn_model(batch_x, w1, b1, w2, b2)
                loss = loss_fn(logits, batch_y)
            gradients = tape.gradient(loss, [w1, b1, w2, b2])
            optimizer.apply_gradients(zip(gradients, [w1, b1, w2, b2]))
            avg_loss += loss/total_batch
            test_logits = nn_model(X_test, w1, b1, w2, b2)
        max_idxs = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
        print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy={test_acc*100:.3f}%")
        if (epoch + 1) % 5 == 0:
            previous = accuracy
            accuracy = test_acc*100
        nn_predict.append(max_idxs.numpy())
    else:
        break
nn_predict = pd.DataFrame(nn_predict, index=['Neural Network Predictions']).transpose()
print(nn_predict)


# In[17]:


output = pd.concat([df_test, rand_for_predict, xgb_predict, nn_predict], axis=1)
print(output.head(5))
# output.to_csv('titanic_output.csv')


# In[18]:


print("Scores for Neural Network: \n"+ classification_report(output['Survived'], output['Neural Network Predictions']))

