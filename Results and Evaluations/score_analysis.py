import pandas as pd

yelpScores = pd.read_csv('/Users/yathartharora/transformer-drg-style-transfer/yelp_bleu_score.csv')
imageScores = pd.read_csv('/Users/yathartharora/transformer-drg-style-transfer/imagecaption_bleu_score.csv')
amazonScores = pd.read_csv('/Users/yathartharora/transformer-drg-style-transfer/amazon_bleu_score.csv')
gyfc_entertainment = pd.read_csv('/Users/yathartharora/transformer-drg-style-transfer/gyfc_entertainment_bleu_score.csv')
gyfc_family = pd.read_csv('/Users/yathartharora/transformer-drg-style-transfer/gyfc_family_bleu_score.csv')

print("*-----YELP DATASET-----*")
print("Mean:",yelpScores['SCORE'].mean())
print("Median:",yelpScores['SCORE'].median())

print("\n")
print("*-----Image Caption DATASET-----*")
print("Mean:",imageScores['SCORE'].mean())
print("Median:",imageScores['SCORE'].median())

print("\n")
print("*-----Amazon DATASET-----*")
print("Mean:",amazonScores['SCORE'].mean())
print("Median:",amazonScores['SCORE'].median())


print("\n")
print("*-----GYFC DATASET-----*")
print("Mean:",gyfc_entertainment['score'].mean())
print("Median:",gyfc_entertainment['score'].median())

print("\n")
print("*-----GYFC DATASET-----*")
print("Mean:",gyfc_family['score'].mean())
print("Median:",gyfc_family['score'].median())