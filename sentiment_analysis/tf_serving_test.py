import tensorflow as tf
import requests

data = {"instances":
            ["I'm surprised I never heard of this before. It had some really funny moments. The attack goats cracked "
             "me up."]}

response = requests.post("http://localhost:8501/v1/models/imdb_bert_wwm_batch12_lr2e-5_epoch2:predict", json=data)

predictions = tf.sigmoid(tf.constant(response.json()["predictions"]))

print(predictions.numpy()[0][0])
