import tensorflow as tf
import tensorflow_text as text
from build_model import build_classifier_model
reloaded_model = tf.saved_model.load("models/imdb_bert_wwm_batch12_lr3e-5_epoch2")


def print_my_examples(inputs, results):
    result_for_printing = \
        [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
         for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


classifier_model = build_classifier_model()

examples = [
    # pos
    "I'm surprised I never heard of this before. It had some really funny moments. The attack goats cracked me up.",
    "Because you'll probably be confused the first time around. It's not a coincidence it stars some of the greatest actors of our time.",
    'Superb, and truly one of the greatest movies of all time.It starts with the screenplay. Adapted from, and very faithful to, an excellent book. The book by Chuck Palahniuk was perfect for a movie: vivid, powerful, challenging, original, unpredictable. Considering how perfectly formed the book already was, the screenplay would have been a doddle.Some very interesting themes are explored - consumerism, class warfare, multiple-personality disorder, male bonding, terrorism and anarchy - without being judgemental. Direction is spot-on. Perfect cinematography, pacing and editing. The twists and nuances of the book are captured perfectly.Edward Norton and Brad Pitt are perfectly cast as the two lead characters, and deliver in spades. Helena Bonham Carter is a strange selection to take on the role of Marla, as she tends to act in Shakespearean dramas and other period pieces. However, despite this, her performance is very convincing.An absolute classic.',
    # neg
    "The film may have a satisfying ending, but the journey to getting there is a little tedious and annoying. It didn't engage me.",
    "This was not as good as the second by a long way. In fact it was bordering on boring. There wasn't enough violence, it moved too slowly and there wasn't enough dancing. It does complete the trilogy except the fourth one that Toxie mentions in this film. Worth making time for if you have seen the first two and want to finish the story. It is a deliberately bad movie.",
    "Chris Rock, who wrote and directed this film, can be very funny. However, this movie wanders all over the place, and when it doesn't work it can be really awful.Why throw in completely over-the-top, unfunny, and highly explicit sex scenes when they seem to come out of left field, and not really congruent with the rest of the story. I much preferred the chemistry between Rock and the superbly talented and beautiful Rosario Dawson, which, I thought, worked really well. Gabrielle Union, J. B. Smoove, and Leslie Jones also added well to the mix here.All in all, as mentioned, the movie is way too choppy, with some really cringe inducing scenes, and overall a disappointment.",
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)
print('Results from the model in memory:')
print_my_examples(examples, original_results)
