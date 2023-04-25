import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def read_file(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            data.append(line.strip())
    return data

yelp = pd.read_csv("/Users/yathartharora/transformer-drg-style-transfer/results/yelp/yelp_all_model_prediction_ref0.csv")
output = read_file("/Users/yathartharora/transformer-drg-style-transfer/T-5/reference_0_predictions_with_beam_search.txt")
# print(yelp.columns)
source = list(yelp.Source)

smoothie = SmoothingFunction().method1

data = pd.DataFrame({'positive': [], 'negative': [], 'score': []})# print(data['HUMAN'])
scores=[]
# data['SCORE'] = pd.Series([0]*len(data))
for i in range(len(source)):
    reference_text = source[i]
    candidate_text = output[i]

    reference_tokens = [sent.split() for sent in reference_text]
    candidate_tokens = candidate_text.split()

    score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    scores.append(score)
    new_row = {'positive': reference_text, 'negative': candidate_text, 'score': score }
    data = data.append(new_row,ignore_index=True)


data.to_csv('yelp_T5.csv')