import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def train_text():
    warnings.filterwarnings('ignore')

    df = pd.read_csv('/Users/DeVante/IdeaProjects/Fake_News_Classifier/test/EXAMPLE_DATA.tsv', delimiter='\t', header=None)
    batch_1 = df[:2000]
    batch_1[1].value_counts()
    print("created batch...")
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    print("model created...")
    tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # pad the sentence length of the 2d matrix. Less accuracy may be occur but
    # performance will improve along with memory usage

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    print("tokenized...")
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    print(attention_mask.shape, '\n')
    print("padding added...")

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    labels = batch_1[1]

    # training model

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    print("training...")

    # grid_search = GridSearchCV(LogisticRegression(), parameters)
    # grid_search.fit(train_features, train_labels)

    # print('best parameters: ', grid_search.best_params_)
    # print('best scrores: ', grid_search.best_score_)

    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    print("Accuracy equals: ", lr_clf.score(test_features, test_labels))

    x = ""
    while x != "stop":
        x = input("press enter to continue:")
        print("test features: ", test_features)
        print("test labels: ", test_labels)
        print("train features: ", test_features)
        print("train labels: ", test_labels)
        print("lr_clf: ", lr_clf)
        print(lr_clf.predict([["the books sold at the store were a really good read. I suggest that you look inside one day and see what they have"],
                              ["The sky is blue"],
                              ["The water is flowing fast but the lava is thundering down on the surface of avatar the last airbender"],
                              ["The Playstation 5 is better than Xbox"],
                              ["Fish have three legs and walk on water. They are known to swallow crocodiles and burn meteors"]]))
    return lr_clf


if __name__ == '__main__':
    train_text()
