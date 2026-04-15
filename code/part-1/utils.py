import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


QWERTY_NEIGHBORS = {
    'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg',
    'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
    'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o', 'q': 'w', 'r': 'et',
    's': 'ad', 't': 'ry', 'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc',
    'y': 'tu', 'z': 'x',
}


def introduce_typo(word):
    if len(word) <= 2:
        return word
    idx = random.randint(0, len(word) - 1)
    char = word[idx].lower()
    if char in QWERTY_NEIGHBORS:
        replacement = random.choice(QWERTY_NEIGHBORS[char])
        if word[idx].isupper():
            replacement = replacement.upper()
        word = word[:idx] + replacement + word[idx + 1:]
    return word


def synonym_replace(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                return name
    return word


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    text = example["text"]
    words = word_tokenize(text)
    new_words = []
    for word in words:
        if not word.isalpha():
            new_words.append(word)
            continue
        r = random.random()
        if r < 0.25:
            new_words.append(synonym_replace(word))
        elif r < 0.50:
            new_words.append(introduce_typo(word))
        else:
            new_words.append(word)

    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
