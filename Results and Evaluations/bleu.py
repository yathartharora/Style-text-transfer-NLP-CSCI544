import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
yelp = pd.read_csv("/Users/yathartharora/transformer-drg-style-transfer/results/amazon/amazon_all_model_prediction_0.csv")

# print(yelp.columns)
features = ['Source','DELETEANDRETRIEVE','HUMAN']


smoothie = SmoothingFunction().method1

data = yelp[features]
# print(data['HUMAN'])
scores=[]
data['SCORE'] = pd.Series([0]*len(data))
for i in range(len(data)):
    reference_text = data['HUMAN'][i]
    candidate_text = data['DELETEANDRETRIEVE'][i]

    reference_tokens = [sent.split() for sent in reference_text]
    candidate_tokens = candidate_text.split()

    score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    scores.append(score)
    data['SCORE'][i] = score


data.to_csv('amazon_bleu_score.csv')
