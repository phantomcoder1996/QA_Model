"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset
import re
import string

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.
        3) characters --> indices,
        4) indices --> characters.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
        charset: A list of all the characters in the dataset in addition to unk
    """
    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}
        self.charset = [PAD_TOKEN, UNK_TOKEN] + list(filter(lambda x:x!=None,set([char if re.match(r"[a-z0-9]+",char) or char in string.punctuation else None for word in self.words[2:] for char in word ]))) #ignore the the unk and pad token from the list of words

        print(self.charset)
        self.charencoding = {char: index for (index, char) in enumerate(self.charset)}
        self.max_word_length = max([len(word) for word in self.words[2:]]) # find the maximum words length
        print("Number of characters = {}".format(len(self.charset)))

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _) in samples:
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        print(len(vocab))
        if vocab_size !=-1:
            top_words = [
                word for (word, _) in
                sorted(vocab.items(), key=lambda x: x[1], reverse=True)
            ][:vocab_size]
        else:
            top_words = [
                word for (word, _) in
                sorted(vocab.items(), key=lambda x: x[1], reverse=True)
            ] 
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)
    
    def numCharacters(self):
        return len(self.charset)


class Tokenizer:
    """
    This class provides three methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words,
        3) List of words --> List of List of character indices.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.max_word_length = self.vocabulary.max_word_length
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]
        self.unk_char_token_id = self.vocabulary.charencoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]
    
    def convert_tokens_to_character_ids(self,tokens):
        """
        Converts tokens to a list of list of character indices padded up to max word length

        Args:
            tokens: a list of words
            max_word_length: maximum word lenght in the vocab , if word lenght is greater than the maximum word length then should truncate, otherwise should pad if not equal

        Returns:
            A list of the actual lengths of each word
            A list of list of character indices (int)
            
            
        """
        seq_char_list = []
        for token in tokens:
            word_char_list = []
            for i in range(self.max_word_length):
                if i<len(token):
                    word_char_list.append(self.vocabulary.charencoding.get(token[i],self.unk_char_token_id))
                else:
                    word_char_list.append(self.vocabulary.charencoding.get(PAD_TOKEN))
            seq_char_list.append(word_char_list)
                    
        return [len(token) for token in tokens], seq_char_list
        # #[
        #     [ self.vocabulary.charencoding.get(token[i], self.unk_char_token_id) if i<len(token) else self.vocabulary.charencoding.get(PAD_TOKEN)
        #     for i in range(self.max_word_length) ] for token in tokens
        # ]




class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self,path):
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()


    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        samples = []
        for elem in self.elems:
            # Unpack the context paragraph. Shorten to max sequence length.
            passage = [
                token.lower() for (token, offset) in elem['context_tokens']
            ]

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for qa in elem['qas']:
                qid = qa['qid']
                question = [
                    token.lower() for (token, offset) in qa['question_tokens']
                ]

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]
                samples.append(
                    (qid, passage, question, answer_start, answer_end)
                )
                
        return samples

    
if __name__ == "__main__":

    squad_train = QADataset(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/datasets/squad_train.jsonl.gz")
    squad_vocab = Vocabulary(squad_train.samples,50000)
    with open(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/vocabulary/squad_train.vocab","w",encoding="utf-8") as f:
        for word in squad_vocab.words:
            f.write(word+"\n")
    squad_dev = QADataset(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/datasets/squad_dev.jsonl.gz")
    squad_dev_vocab = Vocabulary(squad_dev.samples,-1)
    with open(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/vocabulary/squad_dev.vocab","w",encoding="utf-8") as f:
        for word in squad_dev_vocab.words:
            f.write(word+"\n")
    bioasq = QADataset(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/datasets/bioasq.jsonl.gz")
    bioasq_vocab = Vocabulary(bioasq.samples,-1)
    with open(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/vocabulary/bioasq.vocab","w",encoding="utf-8") as f:
        for word in bioasq_vocab.words:
            f.write(word+"\n")

    newsqa_train = QADataset(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/datasets/newsqa_train.jsonl.gz")
    newsqa_train_vocab = Vocabulary(newsqa_train.samples,-1)
    with open(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/vocabulary/newsqa_train.vocab","w",encoding="utf-8") as f:
        for word in newsqa_train_vocab.words:
            f.write(word+"\n")

    newsqa_dev = QADataset(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/datasets/newsqa_dev.jsonl.gz")
    newsqa_dev_vocab = Vocabulary(newsqa_dev.samples,-1)
    with open(r"/Users/reem/Documents/UTAustinMasters/NLP/FinalProject/nlp-qa-finalproj/vocabulary/newsqa_dev.vocab","w",encoding="utf-8") as f:
        for word in newsqa_dev_vocab.words:
            f.write(word+"\n")