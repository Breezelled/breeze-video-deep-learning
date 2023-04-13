import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from const import BERT_EN_WWM_CASED_URL, BERT_EN_CASED_PREPROCESS_URL
from sentiment_analysis_model_train import build_classifier_model

reloaded_model = tf.saved_model.load("imdb_bert_wwm_batch12_r3e-5_epoch2")

tfhub_handle__encoder = BERT_EN_WWM_CASED_URL
tfhub_handle_preprocess = BERT_EN_CASED_PREPROCESS_URL


def print_my_examples(inputs, results):
    result_for_printing = \
        [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
         for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


classifier_model = build_classifier_model()

examples = [
    "I'm surprised I never heard of this before. It had some really funny moments. The attack goats cracked me up.",
    "The film may have a satisfying ending, but the journey to getting there is a little tedious and annoying. It didn't engage me.",
    "Because you'll probably be confused the first time around. It's not a coincidence it stars some of the greatest actors of our time.",
    "I have seen this movie over 10 times. Years later it's still stands strong. It's an awesome movie that I WILL watch again.",
    'Superb, and truly one of the greatest movies of all time.It starts with the screenplay. Adapted from, and very faithful to, an excellent book. The book by Chuck Palahniuk was perfect for a movie: vivid, powerful, challenging, original, unpredictable. Considering how perfectly formed the book already was, the screenplay would have been a doddle.Some very interesting themes are explored - consumerism, class warfare, multiple-personality disorder, male bonding, terrorism and anarchy - without being judgemental. Direction is spot-on. Perfect cinematography, pacing and editing. The twists and nuances of the book are captured perfectly.Edward Norton and Brad Pitt are perfectly cast as the two lead characters, and deliver in spades. Helena Bonham Carter is a strange selection to take on the role of Marla, as she tends to act in Shakespearean dramas and other period pieces. However, despite this, her performance is very convincing.An absolute classic.'
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)
print('Results from the model in memory:')
print_my_examples(examples, original_results)
