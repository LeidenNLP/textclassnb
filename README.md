# Minimal Text-Classification with Naive Bayes

### Example program for the course 'Python for Linguists 2' @ Leiden University. 

This program invokes `scikit-learn` to fit a multinomial Naive Bayes classifier on input texts plus labels, and creates a plot reflecting input feature importances. It can be used to identify, for instance, which words are associated with which sentence types.

## Install

In whichever virtual environment you want this tool to be available (else use `pipx`):

`pip install git+https://github.com/leidennlp/textclassnb`

## Usage

Each line of the input should be a text with a label, like this:

```text
this movie sucks,negative
this movie is pretty great,positive
I really like what the director did here.,positive
I don't dislike it, in fact I would watch it again!,positive
```

But it works just as well if the input comprises, for instance, pre-computed parts of speech:

```text
DET NOUN VERB ADJ NOUN,neutral
NOUN ADP DET NOUN ADV ADV,positive
ADJ ADP PRON NOUN,negative
DET DET NOUN NOUN,positive
PRON NOUN ADP NOUN CCONJ NOUN VERB,negative
ADP DET NOUN,positive
```

Given such data in `text_with_labels.txt`, the following basic usage will print a report, and show a plot:

```bash
$ python multinomialnb.py texts_with_labels.txt
```

Using some command-line options to compensate for class imbalance and filter out (English) stopwords:

```bash
$ cat texts_with_labels.txt | python multinomialnb.py --balanced --stopwords
```

And there are some options to customize the created plot:

```bash
$ cat texts_with_labels.txt | python multinomialnb.py --feature_name "words" --class_name "class" --n_features 20 --saveplot plot.png
```