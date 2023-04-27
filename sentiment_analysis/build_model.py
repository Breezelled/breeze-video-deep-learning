import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from const import BERT_EN_WWM_CASED_URL, BERT_EN_CASED_PREPROCESS_URL, DATASET, BACKUP_BERT_EN_CASED_PREPROCESS_URL, \
    BACKUP_BERT_EN_WWM_CASED_URL

# download dataset
url = DATASET

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')

# remove unused folders to make it easier to load the data
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 12
seed = 42

# create labeled dataset
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    # have test and train dataset already, splitting them to create a validation dataset
    validation_split=0.2,
    subset='validation',
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# quick peek
# for text_batch, label_batch in train_ds.take(1):
#   for i in range(3):
#     print(f'Review: {text_batch.numpy()[i]}')
#     label = label_batch.numpy()[i]
#     print(f'Label : {label} ({class_names[label]})')

# BERT model from tf hub
tfhub_handle_encoder = BERT_EN_WWM_CASED_URL
tfhub_handle_preprocess = BERT_EN_CASED_PREPROCESS_URL

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

# Text inputs need to be transformed to numeric token ids and arranged in several Tensors before being input to BERT.
# bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

# test the preprocessing model on some text
text_test = ['this is such an amazing movie!']


# text_preprocessed = bert_preprocess_model(text_test)
#
# print(f'Keys       : {list(text_preprocessed.keys())}')
# print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
# print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
# print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
# print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

# test BERT before into own model
# bert_model = hub.KerasLayer(tfhub_handle_encoder)

# bert_results = bert_model(text_preprocessed)
#
# print(f'Loaded BERT: {tfhub_handle_encoder}')
# print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
# print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
# print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
# print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

# print("Tensorflow Version:", tf.__version__)
# print("Tensorflow Hub Version:", hub.__version__)
# print("Tensorflow Text Version:", text.__version__)
#
# bert_model = build_classifier_model()
# print(bert_model.summary())