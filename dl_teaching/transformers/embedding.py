from typing import Optional, List
import re


class WordEmbedding:
    """Convert raw string into word embedding."""

    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD"

    def __init__(self, list_of_sentences: Optional[List[str]]) -> None:
        """
        Parameters:
        ----------
        list_of_sentences   : Optional[List[str]]
            the raw input sequences.
        """

        self.token2index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index2token = {v: k for k, v in self.token2index.items()}
        if not list_of_sentences:
            return
        for sentence in list_of_sentences:
            self.add_tokens(self.tokenize(sentence))

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Adds tokens the vocabulary.
        This is the encoding process, using incremental integers to represent the vocabulary.

        Parameters:
        ----------
        tokens   :  List[str]
            the raw input tokens.
        """
        for token in tokens:
            if token not in self.token2index:
                i = len(self.token2index.items())
                self.token2index[token] = i
                self.index2token[i] = token

    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Split on all tokens and punctuation. Optionally adds BOS and EOS tokens.

        Parameters:
        ----------
        sentence   :  str
            the raw input sequence.
        add_special_tokens  :   bool
            a flag to decide whether to add special tokens during tokenization, default to True.

        Returns:
        -------
        tokens  :   List[str]
            the raw tokens from the input sequence.
        """
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converts a string to a list of token indices given the vocabulary

        Parameters:
        ----------
        sentence   :  str
            the raw input sequence.
        add_special_tokens  :   bool
            a flag to decide whether to add special tokens during tokenization, default to True.

        Returns:
        -------
        encoding  :   List[int]
            the encoding of the input sequence.
        """
        tokens = self.tokenize(sentence, add_special_tokens)
        encoding = [self.token2index[token] for token in tokens]
        return encoding

    def batch_encode(
        self, sentences: List[str], padding=True, add_special_tokens: bool = False
    ) -> List[List[int]]:
        """
        Convert a list of string sentences to nested list of token indices. Optionally adds padding & bos+eos tokens

        Parameters:
        ----------
        sentences   :  List[str]
            a batch of raw input sequences.
        padding     :   bool
            a flag to decide whether to pad the sequences to the longest sequence in the batch.
        add_special_tokens  :   bool
            a flag to decide whether to add special tokens during tokenization, default to True.

        Returns:
        -------
        encoding  :   List[List[int]]
            the encoding of the input batch.
        """
        encoding = [self.encode(sentence, add_special_tokens) for sentence in sentences]
        if padding:
            max_length = max([len(tokens) for tokens in encoding])
            encoding = [
                s + ((max_length - len(s)) * [self.token2index[self.PAD]])
                for s in encoding
            ]
        return encoding
