import argparse
import sys

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import compute_sample_weight

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Iterable

MODEL_CLASS = MultinomialNB
CV_FOLDS = 5


def main():

    argparser = argparse.ArgumentParser(description="Text classification with multinomial Naive Bayes, reporting basic stats and a plot of feature importances.")
    argparser.add_argument('file', nargs='?', type=argparse.FileType(mode='r'), default=sys.stdin, help='Input lines consisting of text,category.')
    argparser.add_argument('--save_plot', type=argparse.FileType(mode='w'), default=None)
    argparser.add_argument('--feature_name', type=str, default='Input feature', help='Only used to customize the plot label.')
    argparser.add_argument('--class_name', type=str, default='Class', help='Only used to customize the plot label.')
    argparser.add_argument('--n_features', type=int, default=10, help='Will print and plot only top N most important features.')
    argparser.add_argument('--balanced', action='store_true', help='Whether to balance class labels, for (perhaps) a more interpretable report.')
    argparser.add_argument('--stopwords', action='store_true', help='Whether to filter out (English) stopwords.')

    args = argparser.parse_args()

    texts, labels = read_input(args.file)
    X, y, feature_names, class_labels = prepare_data(texts, labels, args.stopwords)

    model = MODEL_CLASS()
    fit_params = {'sample_weight': compute_sample_weight("balanced", y=y)} if args.balanced else {}

    # for predictions, fit with cross-validation:
    y_predicted = cross_val_predict(model, X, y, cv=CV_FOLDS, fit_params=fit_params)

    # for feature probabilities, fit on ALL data
    model.fit(X, y, **fit_params)
    feature_probs = extract_feature_probs(model, class_labels, feature_names, args.n_features)

    print_report(y, y_predicted, feature_probs, class_labels)
    plot_feature_probs(feature_probs, args.feature_name, args.class_name)

    if args.save_plot:
        plt.savefig(args.save_plot)
        print(f'Plot saved to {args.save_plot}.')
    else:
        print(f'Displaying plot in separate window.')
        plt.show()


def read_input(file: Iterable[str]) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for line in file:
        text, label = line.strip().rsplit(',', maxsplit=1)
        texts.append(text)
        labels.append(label)
    return texts, labels


def prepare_data(texts: Iterable[str], labels: Iterable[str], stopwords: bool):
    vectorizer = CountVectorizer(stop_words='english' if stopwords else None)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    y = np.array(labels)
    class_labels = sorted(set(labels))
    return X, y, feature_names, class_labels


def extract_feature_probs(model, class_labels: list[str], feature_names: list[str], top_n: int) -> pd.DataFrame:
    feature_probs = np.exp(model.feature_log_prob_)
    feature_probs = feature_probs[[(model.classes_ == c).nonzero()[0].squeeze() for c in class_labels]]
    feature_probs_df = pd.DataFrame(feature_probs, index=class_labels, columns=feature_names)
    top_feature_probs_df = feature_probs_df[feature_probs_df.max(axis=0).sort_values()[-top_n:].index]
    return top_feature_probs_df


def plot_feature_probs(feature_probs: pd.DataFrame, feature_name: str, class_name: str) -> None:
    feature_props_melted = (feature_probs
                            .reset_index(names=class_name)
                            .melt(id_vars=class_name, var_name=feature_name, value_name='Probability'))

    plt.figure(figsize=(24, 26))
    sns.barplot(data=feature_props_melted, x=feature_name, y='Probability', hue=class_name)
    plt.title(f'{feature_name} probabilities per {class_name}')


def print_report(y, y_predicted, feature_probs, class_labels):
    titleformat = lambda s: f' {s} '.center(60, '-')

    print(titleformat('Classification Report'))
    print(classification_report(y, y_predicted, target_names=class_labels))

    print(titleformat('Confusion Matrix'))
    conf_matrix = confusion_matrix(y, y_predicted, labels=class_labels)
    print(pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels))
    print()

    print(titleformat('Feature Probabilities Given Class'))
    print(feature_probs.to_string(float_format='{:.2f}'.format))


if __name__ == '__main__':
    main()