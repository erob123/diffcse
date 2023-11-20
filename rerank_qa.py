## Load our cross-encoder. Use fast tokenizer to speed up the tokenization
from sentence_transformers import CrossEncoder
cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)


def cross_score(model_inputs):
    scores = cross_model.predict(model_inputs)
    return scores

model_inputs = [[query, para] for para in results]
scores = cross_score(model_inputs)
#Sort the scores in decreasing order
ranked_results = [{'Para': input, 'Score': score} for input, score in zip(results, scores)]
ranked_results = sorted(ranked_results, key=lambda x: x['Score'], reverse=True)