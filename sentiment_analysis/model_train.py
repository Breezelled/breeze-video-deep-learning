import tensorflow as tf
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from const import BERT_EN_WWM_CASED_URL, BERT_EN_CASED_PREPROCESS_URL, DATASET
import matplotlib.pyplot as plt
import build_model as bm

tf.get_logger().setLevel('ERROR')

# check the model runs with the output of the preprocessing model
classifier_model = bm.build_classifier_model()
# bert_raw_result = classifier_model(tf.constant(text_test))
# print(tf.sigmoid(bert_raw_result))

# because this is a binary classification problem and the model outputs a probability
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# use the same schedule as BERT pre-training
epochs = 2
steps_per_epoch = tf.data.experimental.cardinality(bm.train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
# linear decay of a notional initial learning rate,
# prefixed with a linear warm-up phase over the first 10% of training steps
num_warmup_steps = int(0.1 * num_train_steps)

# In line with the BERT paper, the initial learning rate is smaller for fine-tuning (best of 5e-5, 3e-5, 2e-5)
init_lr = 2e-5

# the "Adaptive Moments" (Adam)
# This optimizer minimizes the prediction loss and does regularization by weight decay (not using moments),
# which is also known as AdamW.
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# loading the BERT model and training
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {bm.tfhub_handle_encoder}')
history = classifier_model.fit(x=bm.train_ds,
                               validation_data=bm.val_ds,
                               epochs=epochs)

# evaluate the model
loss, accuracy = classifier_model.evaluate(bm.test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# plot the training and validation loss, also the training and validation accuracy for comparison
history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("bert_wwm_batch12_lr2e-5_epoch2_loss.svg")
plt.show()

# plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig("bert_wwm_batch12_lr2e-5_epoch2_accuracy.svg")
plt.show()

# save fine-tuned model
dataset_name = 'imdb'
saved_model_path = './{}_bert_wwm_batch12_lr2e-5_epoch2'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)
