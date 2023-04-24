import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



def read_file(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            data.append(line.strip())
    return data

formal = read_file("/Users/yathartharora/transformer-drg-style-transfer/gyfc/formal")
informal = read_file("/Users/yathartharora/transformer-drg-style-transfer/gyfc/informal.nmt_copy")

smoothie = SmoothingFunction().method1
scores=[]
data = pd.DataFrame({'formal': [], 'informal': [], 'score': []})
for i in range(len(formal)):
    reference_text = formal[i]
    candidate_text = informal[i]
    print(reference_text)
    print(candidate_text)

    reference_tokens = [sent.split() for sent in reference_text]
    candidate_tokens = candidate_text.split()

    score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    scores.append(score)

    new_row = {'formal': reference_text, 'informal': candidate_text, 'score': score }
    data = data.append(new_row,ignore_index=True)


data.to_csv('gyfc_family_bleu_score.csv')