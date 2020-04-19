import re

import contractions
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from segtok.tokenizer import split_possessive_markers

from src.utils.stop_words import EN_STOP_WORDS


class TextPreprocessor:

    def __init__(self,
                 lowercase=True,
                 clean_links=True,
                 clean_punctuation=True,
                 expand_contractions=False,
                 remove_stop_words=True,
                 process_numbers='remove',
                 normalization_type='lemma',
                 clean_mentions=False,
                 tokenizer=None,
                 stop_words=None):
        self._lowercase = lowercase
        self._clean_links = clean_links
        self._expand_contractions = expand_contractions
        self._clean_punc = clean_punctuation
        self._remove_stop_words = remove_stop_words
        self._process_numbers = process_numbers
        self._normaliztion_type = normalization_type
        self._clean_mentions = clean_mentions

        self.tokenizer = tokenizer or TweetTokenizer().tokenize
        self.stop_words = stop_words or EN_STOP_WORDS

        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

        self.url_rgx = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.pic_rgx = re.compile(r'(pic\.twitter\.com\/[a-zA-Z0-9]+)')
        self.punc_rgx = re.compile(r'[^\w\s]')
        self.numbers_rgx = re.compile(r'\d')
        self.mention_rgx = re.compile(r'@[^\s]+')

        self._make_pipeline()

    def _make_pipeline(self):
        self.pipeline = [
            lambda text: text.lower() if self._lowercase else text,
            self.clean_links if self._clean_links else lambda text: text,
            self.clean_mentions if self._clean_mentions else lambda text: text,
            self.expand_contractions
            if self._expand_contractions else lambda text: text,
            self._get_numbers_processor(),
            self.tokenizer,
            split_possessive_markers,
            self._get_normalizer(),
            self.remove_stop_words
            if self._remove_stop_words else lambda text: text,
            self.clean_punctuation if self._clean_punc else lambda text: text,
        ]

    def clean_links(self, text):
        text = self.url_rgx.sub(' ', text).strip()
        text = self.pic_rgx.sub(' ', text).strip()
        return text

    def clean_punctuation(self, text):
        if isinstance(text, str):
            text = self.punc_rgx.sub(' ', text).strip()
            return text
        elif isinstance(text, list):
            tokens = []
            for token in text:
                token = self.punc_rgx.sub(' ', token).strip()
                if token:
                    tokens.append(token)
            return tokens
        else:
            raise TypeError(
                'Incorrect input type. Should be str or list of str')

    def expand_contractions(self, text):
        text = contractions.fix(text)
        text = text.lower() if self._lowercase else text
        return text

    def remove_stop_words(self, tokens):
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def clean_numbers(self, text):
        text = self.numbers_rgx.sub(r'', text)
        return text

    def replace_numbers(self, text):
        text = self.numbers_rgx.sub(r'0', text)
        return text

    def _get_numbers_processor(self):
        if self._process_numbers == 'remove':
            return self.clean_numbers
        elif self._process_numbers == 'replace':
            return self.replace_numbers
        elif self._process_numbers is None:
            return lambda x: x
        else:
            msg = 'Invalid argument. process_numbers argument should be ' \
                  'remove, replace or None'
            raise ValueError(msg)

    def _get_normalizer(self):
        if self._normaliztion_type == 'stem':
            return self.stem_words
        elif self._normaliztion_type == 'lemma':
            return self.lemmatize_words
        elif self._normaliztion_type is None:
            return lambda x: x
        else:
            msg = 'Invalid argument. Normalization type should be stem, ' \
                  'lemma or None'
            raise ValueError(msg)

    def stem_words(self, tokens):
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def lemmatize_words(self, tokens):
        tokens = [
            self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token))
            for token in tokens
        ]
        return tokens

    def get_wordnet_pos(self, token):
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

        tag = pos_tag([token])[0][1][0].upper()

        return tag_dict.get(tag, wordnet.NOUN)

    def clean_mentions(self, text):
        text = self.mention_rgx.sub('', text)
        return text

    def __call__(self, text):
        for processor in self.pipeline:
            text = processor(text)
        return text
