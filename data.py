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
#from memory_profiler import profile
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
    def __init__(self, samples, vocab_size,filepath=None,load_from_file=False):
        if load_from_file:
            self.words = self._load_vocab_from_file(filepath)
        else:
            # just create the vocabulary from the training data
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
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)

    def _load_vocab_from_file(self,filepath):
        """
        loads the vocabulary words from file
        
        Args:
            filepath: path of the vocabfile

        Returns:
            words: list of words in file
        """
        words = []
        with open(filepath,"r",encoding="utf-8") as f:
            words = [line.strip() for line in f.readlines()]
        return [PAD_TOKEN, UNK_TOKEN] + words

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
    def __init__(self, args, path):
        self.args = args
        print("Number of data sets provided = {}".format(len(path)))
        print(path)
        self.elems = []
        for datapath in path:
            meta, elems = load_dataset(datapath)
            self.elems.extend(elems)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

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
            ][:self.args.max_context_length]

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for qa in elem['qas']:
                qid = qa['qid']
                question = [
                    token.lower() for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]
                samples.append(
                    (qid, passage, question, answer_start, answer_end)
                )
                
        return samples
    
    
    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)
        
        # passages = []
        # questions = []
        # start_positions = []
        # end_positions = []
        # passages_char = []
        # questions_char = []
        # passage_lens = []
        # question_lens = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, answer_start, answer_end = self.samples[idx]

            # Convert words to tensor.
            passage_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(passage)
            )
            question_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(question)
            )
            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)
            #import pdb;pdb.set_trace()
            # convert word characters to tensor
            passage_word_lens, passage_chars = self.tokenizer.convert_tokens_to_character_ids(passage) # plen, plen * max_word_len
           
            passage_char_ids = torch.tensor(passage_chars)
            passage_word_len = torch.tensor(passage_word_lens) #plen

            question_word_lens, question_chars = self.tokenizer.convert_tokens_to_character_ids(question) # qlen, qlen * max_word_len
            question_char_ids = torch.tensor(question_chars) 
            question_word_len = torch.tensor(question_word_lens) #qlen

            # # Store each part in an independent list.
            # passages.append(passage_ids)
            # questions.append(question_ids)
            # start_positions.append(answer_start_ids)
            # end_positions.append(answer_end_ids)
            # passages_char.append(passage_char_ids)
            # questions_char.append(question_char_ids)
            # passage_lens.append(passage_word_len)
            # question_lens.append(question_word_len)
            yield (passage_ids,question_ids,answer_start_ids,answer_end_ids,passage_char_ids,question_char_ids,passage_word_len,question_word_len)

        #return zip(passages, questions, start_positions, end_positions,passages_char,questions_char,passage_lens,question_lens)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            passages_char = []
            questions_char = []
 

            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                passages_char.append(current_batch[ii][4])
                questions_char.append(current_batch[ii][5])
                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]
            
                # p_wlens.append(current_batch[ii][6])
                # q_wlens.append(current_batch[ii][7])
                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages = torch.zeros(bsz, max_passage_length)
            padded_questions = torch.zeros(bsz, max_question_length)

            # create a vector for the word characters, use the pad character index for words that are pad tokens
            # Assume character pad index = 0
            max_word_length = current_batch[0][4].size(-1)
            padded_passages_char = torch.zeros(bsz,max_passage_length,max_word_length)
            padded_questions_char = torch.zeros(bsz,max_question_length,max_word_length)



            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages, questions,passages_char,questions_char)):
                passage, question , passage_char, question_char = passage_question
                padded_passages[iii][:len(passage)] = passage
                padded_questions[iii][:len(question)] = question
                padded_passages_char[iii][:passage_char.size(0)][:] = passage_char
                padded_questions_char[iii][:question_char.size(0)][:] = question_char
                # padded_passage_word_lens[iii][:len(passage)][:] =
                # padded_question_word_lens[iii][:len(question)] = q_wlen


            # Create an input dictionary
            batch_dict = {
                'passages': cuda(self.args, padded_passages).long(),
                'questions': cuda(self.args, padded_questions).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long(),
                'passages_char': cuda(self.args,padded_passages_char).long(),
                'questions_char': cuda(self.args,padded_questions_char).long()
                # 'passages_word_lens': cuda(self.args,padded_passage_word_lens).long(),
                # 'questions_word_lens': cuda(self.args,padded_question_word_lens).long()

            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
