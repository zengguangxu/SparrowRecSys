import tensorflow as tf
from tensorflow.python.framework import dtypes

train_data_list = [
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-01.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-02.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-03.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-04.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-05.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-06.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-07.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-08.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-09.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-10.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-11.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-12.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-13.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-14.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-15.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-16.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-17.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-18.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-19.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-20.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-21.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-22.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-23.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-24.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-25.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-26.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-27.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-28.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-29.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-30.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-01-31.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-01.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-02.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-03.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-04.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-05.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-06.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-07.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-08.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-09.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-10.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-11.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-12.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-13.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-14.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-15.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-16.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-17.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-18.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-19.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-20.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-21.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-22.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-23.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-24.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-25.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-26.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-27.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-28.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-02-29.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-01.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-02.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-03.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-04.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-05.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-06.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-07.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-08.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-09.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-10.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-11.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-12.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-13.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-14.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-15.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-16.csv',
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-17.csv'
]

test_data_list = [
    '/Users/king.zeng/IdeaProjects/data/king_rec/datas/data-2024-03-18.csv'
]

# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=1024,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset


# split as test dataset and training dataset
train_dataset = get_dataset(train_data_list)
test_dataset = get_dataset(test_data_list)

# movie id embedding feature
pid_col = tf.feature_column.categorical_column_with_hash_bucket(key='pid', dtype=dtypes.int32, hash_bucket_size=200000)
pid_emb_col = tf.feature_column.embedding_column(pid_col, 20)

lc_col = tf.feature_column.categorical_column_with_hash_bucket(key='lc', dtype=dtypes.int32, hash_bucket_size=1000)
lc_emb_col = tf.feature_column.embedding_column(lc_col, 8)

c2_col = tf.feature_column.categorical_column_with_hash_bucket(key='c2', dtype=dtypes.int32, hash_bucket_size=1000)
c2_emb_col = tf.feature_column.embedding_column(c2_col, 8)

w_col = tf.feature_column.categorical_column_with_hash_bucket(key='w', dtype=dtypes.int32, hash_bucket_size=100)
w_emb_col = tf.feature_column.embedding_column(w_col, 8)

sc_col = tf.feature_column.categorical_column_with_hash_bucket(key='sc', dtype=dtypes.int32, hash_bucket_size=100)
sc_emb_col = tf.feature_column.embedding_column(sc_col, 8)

pc_col = tf.feature_column.categorical_column_with_hash_bucket(key='pc', dtype=dtypes.int32, hash_bucket_size=100)
pc_emb_col = tf.feature_column.embedding_column(pc_col, 8)

ct_col = tf.feature_column.categorical_column_with_hash_bucket(key='ct', dtype=dtypes.string, hash_bucket_size=1000)
ct_emb_col = tf.feature_column.embedding_column(ct_col, 8)


# user id embedding feature
cid_col = tf.feature_column.categorical_column_with_hash_bucket(key='cid', dtype=dtypes.int32, hash_bucket_size=100)
cid_emb_col = tf.feature_column.embedding_column(cid_col, 8)

ts_col = tf.feature_column.categorical_column_with_hash_bucket(key='ts', dtype=dtypes.int32, hash_bucket_size=1000)
ts_emb_col = tf.feature_column.embedding_column(ts_col, 8)

# event_col = tf.feature_column.categorical_column_with_identity(key='event', num_buckets=10, default_value=0)
event_col = tf.feature_column.categorical_column_with_hash_bucket(key='event', dtype=dtypes.int32, hash_bucket_size=10)
event_emb_col = tf.feature_column.embedding_column(event_col, 4)

user_col = tf.feature_column.categorical_column_with_hash_bucket(key='uid', dtype=dtypes.int32, hash_bucket_size=500000)
user_emb_col = tf.feature_column.embedding_column(user_col, 20)

# sex_col = tf.feature_column.categorical_column_with_identity(key='sex', num_buckets=5, default_value=0)
sex_col = tf.feature_column.categorical_column_with_hash_bucket(key='sex', dtype=dtypes.int32, hash_bucket_size=5)
sex_emb_col = tf.feature_column.embedding_column(sex_col, 4)

# define input for keras model
inputs = {
    'pid': tf.keras.layers.Input(name='pid', shape=(), dtype='int32'),
    'uid': tf.keras.layers.Input(name='uid', shape=(), dtype='int32'),
    'ts': tf.keras.layers.Input(name='ts', shape=(), dtype='int32'),
    'event': tf.keras.layers.Input(name='event', shape=(), dtype='int32'),
    'ct': tf.keras.layers.Input(name='ct', shape=(), dtype='string'),
    'lc': tf.keras.layers.Input(name='lc', shape=(), dtype='int32'),
    'w': tf.keras.layers.Input(name='w', shape=(), dtype='int32'),
    'sc': tf.keras.layers.Input(name='sc', shape=(), dtype='int32'),
    'pc': tf.keras.layers.Input(name='pc', shape=(), dtype='int32'),
    'c2': tf.keras.layers.Input(name='c2', shape=(), dtype='int32'),
    'cid': tf.keras.layers.Input(name='cid', shape=(), dtype='int32'),
    'sex': tf.keras.layers.Input(name='sex', shape=(), dtype='int32'),
}

item_feature_columns = [
    pid_emb_col,
    lc_emb_col,
    c2_emb_col,
    w_emb_col,
    sc_emb_col,
    pc_emb_col,
    ct_emb_col
]

user_feature_columns = [
    cid_emb_col,
    ts_emb_col,
    event_emb_col,
    user_emb_col,
    sex_emb_col
]

# neural cf model arch two. only embedding in each tower, then MLP as the interaction layers
def neural_cf_model_1(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    interact_layer = tf.keras.layers.concatenate([item_tower, user_tower])
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
    neural_cf_model = tf.keras.Model(feature_inputs, output_layer)
    return neural_cf_model


# neural cf model arch one. embedding+MLP in each tower, then dot product layer as the output
def neural_cf_model_2(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    count = 1
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu', name="item_tower_" + str(count))(item_tower)
        count += 1

    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    count = 1
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu', name="user_tower_" + str(count))(user_tower)
        count += 1

    output = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    for num_nodes in hidden_units:
        output = tf.keras.layers.Dense(num_nodes, activation='relu')(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    neural_cf_model = tf.keras.Model(feature_inputs, output)
    return neural_cf_model


# neural cf model architecture
hidden_units = [15, 10, 10]
model = neural_cf_model_2(inputs, item_feature_columns, user_feature_columns, hidden_units)

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# 输出模型网络结构
# tf.keras.utils.plot_model(model, to_file="./NeuralCF.png", show_shapes=True)

# model.summary()

item_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(name='item_tower_'+str(len(hidden_units))).output)
item_model.summary()
# 输出模型网络结构
tf.keras.utils.plot_model(item_model, to_file="./NeuralCF_Item_king.png", show_shapes=True)

user_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(name='user_tower_'+str(len(hidden_units))).output)
user_model.summary()
# 输出模型网络结构
tf.keras.utils.plot_model(user_model, to_file="./NeuralCF_User_king.png", show_shapes=True)


# train the model
model.fit(train_dataset, epochs=3)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))

tf.keras.models.save_model(
    model,
    "file:///Users/king.zeng/IdeaProjects/SparrowRecSys/src/main/resources/webroot/modeldata/neuralcf/002",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
