import tensorflow as tf

reloaded_model = tf.saved_model.load("models/din_1000/export/final/1682047735")

examples = [
    {  # suitable
        "user_id": 5,
        "liked_sequence": "27705|40148",
        "interest_type": "Action|Drama|Thriller|Crime",
        "info_id": 172591,
        "type": "Crime|Drama|Thriller",
        "favorited_sequence": "27705|40148",
        "gender": "M"
    },
    {  # suitable
        "user_id": 5,
        "liked_sequence": "27705|40148",
        "interest_type": "Action|Drama|Thriller|Crime",
        "info_id": 7153,
        "type": "Action|Adventure|Drama",
        "favorited_sequence": "27705|40148",
        "gender": "M"
    },
    {  # in sequence
        "user_id": 5,
        "liked_sequence": "27705|40148",
        "interest_type": "Action|Drama|Thriller|Crime",
        "info_id": 27705,
        "type": "Action|Drama|Thriller",
        "favorited_sequence": "27705|40148",
        "gender": "M"
    },
    {  # in sequence
        "user_id": 5,
        "liked_sequence": "27705|40148",
        "interest_type": "Action|Drama|Thriller|Crime",
        "info_id": 40148,
        "type": "Action|Crime|Drama",
        "favorited_sequence": "27705|40148",
        "gender": "M"
    },
    {  # unsuitable
        "user_id": 5,
        "liked_sequence": "27705|40148",
        "interest_type": "Action|Drama|Thriller|Crime",
        "info_id": 141994,
        "type": "Comedy|Family",
        "favorited_sequence": "27705|40148",
        "gender": "M"
    },
    {  # unsuitable
        "user_id": 5,
        "liked_sequence": "27705|40148",
        "interest_type": "Action|Drama|Thriller|Crime",
        "info_id": 4051,
        "type": "Horror",
        "favorited_sequence": "27705|40148",
        "gender": "M"
    }
]

input_dict = {}
for key in examples[0].keys():
    # put all the examples' values for the same key into a tensor
    input_dict[key] = tf.stack([example[key] for example in examples])

signature_key = 'serving_default'
model = reloaded_model.signatures[signature_key]
reloaded_results = model(**input_dict)

print('Results from the saved model:')
print(reloaded_results['y'].numpy())

