import re
import csv
import emoji
import pandas
import random
import string
import numpy as np
import pandas as pd

from re import Pattern
from nltk.corpus import wordnet
from collections import defaultdict
from utilities.utils import config, hashtags_dir, input_dir, read_json, write_json


def load_n_words(n, filename):
    """
    Loads most frequent n words
    :param n: number of words to load
    :param filename: name of file to save words to
    :return:
    """
    with open(f'{hashtags_dir}/unigram_freq.csv', 'r', encoding='utf-8') as f:
        words = [tuple(row.replace('\n', '').split(',')) for row in f.readlines()][1:]
        words = [(k, int(v) / 1_000_000_000_000) for k, v in words]
        words.sort(key=lambda x: x[1], reverse=True)
        words = words[:n]

    with open(filename, 'w', encoding='utf-8') as f:
        for k, _ in words:
            f.write(k)
            f.write('\n')


def get_synonyms(word):
    """
    Get synonyms of a word.
    :param word: word to fetch synonyms for
    :return: synonyms
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)


def min_all(xs, key=None):
    """

    :param xs:
    :param key:
    :return:
    """
    if not xs:
        return []
    min_value = key(min(xs, key=key)) if key is not None else min(xs)
    return [x for x in xs if x == min_value] if key is None else [x for x in xs if key(x) == min_value]


def load_words(word_source):
    with open(word_source, 'r', encoding='utf-8') as f:
        return list(set([line.lower().replace('\n', '') for line in f.readlines()]))


def get_chars(words):
    chars = []
    for word in words:
        for c in word:
            chars.append(c)

    chars = set(chars)
    chars = {c: i for i, c in enumerate(chars)}
    return chars


def load_frequencies():
    with open(f'{hashtags_dir}/unigram_freq.csv') as f:
        reader = csv.reader(f, delimiter=',')
        data = [x for x in reader]
    total = sum([int(freq) for _, freq in data[1:]])
    return {word: int(freq) / total for word, freq in data[1:]}


# noinspection RegExpSimplifiable
class PreProcessing:
    def __init__(self, df, word_source=f'{hashtags_dir}/50_000_words.txt', silent=False):
        self.df: pandas.DataFrame = df.copy()
        self.hashtag_counter = 0
        self.words = load_words(word_source)
        self.chars = get_chars(self.words)
        self.seed = int(config['RANDOM_SEED'])
        self.apply = lambda f, dff: dff.progress_apply(f) if not silent else dff.apply(f)
        self.silent = silent

    def convert_html_entities(self):
        """
        Converts HTML to ASCII
        :return: self
        """
        if not self.silent:
            print('Converting HTML Entities')

        known_tags = ['&amp;', '&quot;', '&gt;', '&lt;']
        unknown_tags = []
        entity_regex = re.compile(r'&[^;]+;')

        def check_unknown(x: str):
            """
            Checks if unknown HTML entity and adds to unknown list
            :param x: text containing entities
            :return:
            """
            entities = re.findall(entity_regex, x)
            for entity in entities:
                if entity not in known_tags:
                    unknown_tags.append(entity)

        def convert(x: str) -> str:
            """
            Converts known HTML tags, [&, ", <, >]
            :param x: text to convert
            :return: converted text
            """
            return x.replace('&amp;', '&').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>')

        self.df['text'].apply(check_unknown)
        self.df['text'] = self.apply(convert, self.df['text'])
        if unknown_tags:
            print(f'Unknown Tags: {unknown_tags}')
        return self

    # noinspection DuplicatedCode
    def emojis(self):
        """
        Resolves emojis and emoticons
        :return: self
        """
        if not self.silent:
            print('Converting Emojis')
        smileys = read_json(filename=f'{input_dir}/smileys.json')
        regex = re.compile(r'|'.join([re.escape(key) for key in smileys.keys()]))
        link_regex = re.compile(r'(?:https|http)?://(?:www\.)?[-a-zA-Z\d@:%._+~#=]'
                                r'{1,256}\.[a-zA-Z\d()]{1,6}\b[-a-zA-Z\d()@:%_+.~#?&/=]*')

        def handle_smiley(m, spans):
            """
            Resolves emoticons, avoiding links
            :param m: identified emoticon
            :param spans: spans of links to avoid
            :return: resolved emoticon or raw text if link
            """
            span = m.span()
            span = {i_ for i_ in range(span[0], span[1])}
            if len(set(span).intersection(spans)) != 0:
                return m.group()
            return smileys[m.group()]

        def handle_emojis(text: str) -> str:
            """
            Resolves emojis and emoticons in text
            :param text: raw text
            :return: resolved text
            """
            spans = [x.span() for x in re.finditer(link_regex, text)]
            spans = {x for span in spans for x in [i_ for i_ in range(span[0], span[1])]}

            text = re.subn(regex, lambda m: handle_smiley(m, spans), text)[0]
            text = emoji.demojize(text, delimiters=(' ', ' '))
            return text

        self.df['text'] = self.apply(handle_emojis, self.df['text'])
        return self

    # noinspection DuplicatedCode
    def strip_emojis(self):
        """
        Strips emojis and emoticons
        :return: self
        """
        if not self.silent:
            print('Stripping Emojis')
        smileys = read_json(filename=f'{input_dir}/smileys.json')
        escaped_keys = [re.escape(key) for key in smileys.keys()]
        escaped_keys = [f'{key}(?![a-zA-Z])' for key in escaped_keys]
        regex = re.compile(r'|'.join(escaped_keys))
        delimiters = [('¦', re.compile(r'¦[^℈]*℈')), ('҂', re.compile(r'҂[^℈]℈'))]
        link_regex = re.compile(r'(?:https|http)?://(?:www\.)?[-a-zA-Z\d@:%._+~#=]'
                                r'{1,256}\.[a-zA-Z\d()]{1,6}\b[-a-zA-Z\d()@:%_+.~#?&/=]*')

        def handle_smiley(m, spans):
            """
            Resolves emoticons, avoiding links
            :param m: identified emoticon
            :param spans: spans of links to avoid
            :return: resolved emoticon or raw text if link
            """
            span = m.span()
            span = {i_ for i_ in range(span[0], span[1])}
            if len(set(span).intersection(spans)) != 0:
                return m.group()
            return smileys[m.group()]

        def strip(text):
            """
            Strips emojis and emoticons from text
            :param text: text to strip
            :return: stripped text
            """
            spans = [x.span() for x in re.finditer(link_regex, text)]
            spans = {x for span in spans for x in [i_ for i_ in range(span[0], span[1])]}
            text = re.subn(regex, lambda m: handle_smiley(m, spans), text)[0]
            i = 0 if delimiters[0][0] not in text else 1

            text = emoji.demojize(text, delimiters=(delimiters[i][0], '℈'))
            return re.subn(delimiters[i][1], '', text)[0]

        self.df['text'] = self.apply(strip, self.df['text'])
        return self

    def strip_mentions(self):
        """
        Strips mentions
        :return: self
        """
        if not self.silent:
            print('Stripping Mentions')
        regex = re.compile(r'\B@([\w\-_]+)', flags=re.I | re.M)
        whitespace_regex = re.compile(r' {2,}')
        self.df['text'] = self.apply(lambda x: re.subn(whitespace_regex, ' ', re.subn(regex, ' ', x)[0])[0],
                                     self.df['text'])
        return self

    def _build_seq(self, seq, rr, max_key, acc):
        """
        Recursively finds all contiguous sequences possible for a hashtag
        :param seq: current list of positions
        :param rr: map of start positions and words
        :param max_key: maximum length of sequence
        :param acc: final list of sequences
        :return: all possible sequences
        """
        new_seq = []
        for s in seq:
            next_key = s[-1][2] + 1

            if next_key == max_key + 1:
                acc.append(s)
                continue

            for n in rr[next_key]:
                new_seq.append(s + [n])
        if not new_seq:
            return acc
        return self._build_seq(new_seq, rr, max_key, acc)

    def _contextualise_hashtag(self, hashtag, use_frequencies=False):
        """
        Contextualise individual hashtag
        :param hashtag: hashtag to resolve
        :param use_frequencies: whether to use unigram frequency or word length as optimiser
        :return: resolved hashtag
        """
        if hashtag in self.words:
            return hashtag

        xs = AhoCorasick(self.words, self.chars).search_words(hashtag)
        xs = [(w, i, i + len(w) - 1) for w, ii in xs.items() for i in
              ii]  # convert possible words to (word, start, end)
        xs = {x: [y for y in xs if y[1] == x] for x in set(map(lambda x: x[1], xs))}  # map of start positions and words

        # sort by end position
        for v in xs.values():
            v.sort(key=lambda x: x[2])

        # all possible contiguous sequences
        seq = self._build_seq([[x] for x in xs[min(xs.keys())]], xs, max(xs.keys()), [])
        seq = [s for s in seq if s[0][1] == 0 and s[-1][2] == len(hashtag)]

        if use_frequencies:
            frequencies = load_frequencies()
            seq = min_all(seq, key=lambda x: sum([frequencies[y] for y in x]))
        else:
            # select the option with the fewest words and the fewest single letter words and convert to words
            seq = min_all(seq, key=lambda x: len(x))

        if not seq:
            return hashtag

        seq = [s[0] for s in min(seq, key=lambda x: [len(y[0]) for y in x])]

        if not seq:
            return hashtag

        return ' '.join(seq)  # join to context string

    def strip_hashtags(self):
        """
        Strips hashtags
        :return: self
        """
        if not self.silent:
            print('Stripping Hashtags')

        hashtag_regex = re.compile(r'(?:^|\B)#(?![\d_]+\b)([a-zA-Z0-9_]{1,30})(?:\b|\r)')

        self.df['text'] = self.apply(lambda x: re.subn(hashtag_regex, '', x)[0], self.df['text'])
        return self

    def contextualise_hashtags(
        self,
        cache_source: str = f'{hashtags_dir}/unigram_hashtags_50_000.json',
        limit: int = None,
        use_frequencies=False
    ):
        """
        Contextualises hashtags
        :param cache_source: source of cached, resolved hashtags
        :param limit: optional limit on amount of data contextualised
        :param use_frequencies: whether to use unigram frequency or word length as optimiser
        :return: self
        """
        if not self.silent:
            print('Contextualising Hashtags')

        cached_hashtags = read_json(filename=cache_source)

        uppercase_regex = re.compile(r'^[A-Z\d]+$')
        split_uppercase_regex = re.compile(r'[A-Z][^A-Z]*')
        pascal_case_regex = re.compile(r'^((([A-Z][a-z\d]+)|[A-Z]|\d)+)$')
        hashtag_regex = re.compile(r'(?:^|\B)#(?![\d_]+\b)([a-zA-Z0-9_]{1,30})(?:\b|\r)')

        def format_hashtag(x):
            """
            Matches and contextualises hashtags
            :param x: text containing hashtags
            :return: contextualised text
            """
            x = str(x)
            x = x[1:]

            if cached_hashtags.get(x, None) is not None:
                return cached_hashtags[x]

            if re.match(uppercase_regex, x):
                return x

            if re.match(pascal_case_regex, x):
                return ' '.join(re.findall(split_uppercase_regex, x))

            contextualised = self._contextualise_hashtag(x, use_frequencies=use_frequencies)
            cached_hashtags[x] = contextualised

            if self.hashtag_counter % 100 == 0:
                write_json(obj=cached_hashtags, filename=cache_source)
            self.hashtag_counter += 1

            return contextualised

        def format_text(x):
            """
            Formats text by contextualising hashtags
            :param x: text to format
            :return: formatted text
            """
            return re.subn(hashtag_regex, lambda m: format_hashtag(m.group()), x)[0]

        limited = self.df['text'].head(limit) if limit is not None else self.df['text']
        self.df['text'] = self.apply(format_text, limited)

        write_json(obj=cached_hashtags, filename=cache_source)

        return self

    def strip_newlines(self):
        """
        Strips newlines
        :return: self
        """
        if not self.silent:
            print('Stripping Newlines')
        self.df['text'] = self.apply(lambda x: x.replace('\n', ' '), self.df['text'])
        return self

    def strip_links(self):
        """
        Strips links
        :return: self
        """
        if not self.silent:
            print(f'Stripping URLs')
        regex = re.compile(r'(?:https|http)?://(?:www\.)?[-a-zA-Z\d@:%._+~#=]'
                           r'{1,256}\.[a-zA-Z\d()]{1,6}\b[-a-zA-Z\d()@:%_+.~#?&/=]*')

        def format_text(x):
            """
            Handler for stripping links
            :param x: text to strip
            :return: stripped text
            """
            return re.subn(regex, ' ', x)[0]

        self.df['text'] = self.apply(format_text, self.df['text'])

        return self

    def augment_dataset(self, n: int, reset_index: bool = False):
        """
        Augments the dataset through synonym replacement
        :param n: number of augmentations to make
        :param reset_index: whether to reset the index after augmentation, default is False
        :return: self
        """
        def augment_word(
            word: str,
            lowercase_regex: Pattern[str],
            uppercase_regex: Pattern[str],
            punctuation_regex: Pattern[str],
            rng: np.random.Generator
        ):
            """
            Finds synonyms to replace word
            :param word: word to replace
            :param lowercase_regex: regex for identifying lower case words
            :param uppercase_regex: regex for identifying upper case words
            :param punctuation_regex: regex for identifying punctuation
            :param rng: random generator for repeatability
            :return: synonym for word if present
            """
            if word.isupper():
                synonyms = get_synonyms(word.lower())
                if synonyms:
                    return rng.choice(synonyms).capitalize()
            elif re.fullmatch(lowercase_regex, word):
                synonyms = get_synonyms(word.lower())
                if synonyms:
                    return rng.choice(synonyms)
            elif re.fullmatch(uppercase_regex, word):
                synonyms = get_synonyms(word.lower())
                if synonyms:
                    return rng.choice(synonyms).upper()
            elif re.fullmatch(punctuation_regex, word):
                stripped = word.strip(string.punctuation)
                synonyms = get_synonyms(stripped)
                if synonyms:
                    return f'{rng.choice(synonyms)}{word[len(stripped):]}'
            return word

        def augment(x: str) -> str:
            """
            Augments text through synonym replacement
            :param x: text to augment
            :return: augmented text
            """
            words = x.split(' ')
            subset = {word: word.lower() for word in words if not word.isupper() and
                      all([char in 'qwertyuiopasdfghjklzxcvbnm-' for char in word])}
            if subset == {}:
                return x

            size_rng = random.Random(self.seed)
            rng = np.random.default_rng(self.seed)

            n_replace = max(min(len(words), 3), size_rng.randrange(0, len(words)))
            mask = rng.choice(len(words), replace=False, size=n_replace)

            if n_replace == 0:
                return x

            lowercase_regex = re.compile(r'(?:[a-z]+(?:-[a-z])?)+')
            uppercase_regex = re.compile(r'(?:[A-Z]+(?:-[A-Z])?)+')
            punctuation_regex = re.compile(r'(?:[a-zA-Z]+(?:-[a-zA-Z])?)+[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]*')
            words = [augment_word(word, lowercase_regex, uppercase_regex, punctuation_regex, rng) if i in mask else
                     word for i, word in enumerate(words)]

            return ' '.join(words)

        dfs = [self.df]
        for j in range(n):
            df = self.df.copy()
            df['text'] = self.apply(augment, df['text'])
            dfs.append(df)
        self.df = pd.concat(dfs)

        if reset_index:
            self.df.reset_index(drop=True, inplace=True)
        return self


class AhoCorasick:
    """
    Adapted code from https://www.geeksforgeeks.org/aho-corasick-algorithm-pattern-searching/
    """
    def __init__(self, words, chars):
        self.max_states = sum([len(word) for word in words])
        self.max_characters = len(chars)
        self.out = [0] * (self.max_states + 1)
        self.fail = [-1] * (self.max_states + 1)
        self.goto = [[-1] * self.max_characters for _ in range(self.max_states + 1)]
        self.char_map = chars

        for i in range(len(words)):
            words[i] = words[i].lower()

        self.words = words
        self.states_count = self.__build_matching_machine()

    def __build_matching_machine(self):
        states = 1
        k = len(self.words)

        for i in range(k):
            word = self.words[i]
            current_state = 0

            # Process all the characters of the current word
            for character in word:
                ch = self.char_map[character]
                if self.goto[current_state][ch] == -1:
                    self.goto[current_state][ch] = states
                    states += 1

                current_state = self.goto[current_state][ch]

            self.out[current_state] |= (1 << i)

        for ch in range(self.max_characters):
            if self.goto[0][ch] == -1:
                self.goto[0][ch] = 0

        queue = []
        for ch in range(self.max_characters):
            if self.goto[0][ch] != 0:
                self.fail[self.goto[0][ch]] = 0
                queue.append(self.goto[0][ch])

        while queue:
            # noinspection PyUnresolvedReferences
            state = queue.pop(0)
            for ch in range(self.max_characters):
                if self.goto[state][ch] != -1:
                    failure = self.fail[state]
                    while self.goto[failure][ch] == -1:
                        failure = self.fail[failure]

                    failure = self.goto[failure][ch]
                    self.fail[self.goto[state][ch]] = failure
                    self.out[self.goto[state][ch]] |= self.out[failure]
                    queue.append(self.goto[state][ch])

        return states

    def __find_next_state(self, current_state, next_input):
        answer = current_state
        ch = self.char_map[next_input]

        while self.goto[answer][ch] == -1:
            answer = self.fail[answer]

        return self.goto[answer][ch]

    def search_words(self, text):
        current_state = 0
        text = text.lower()
        result = defaultdict(list)

        for i in range(len(text)):
            current_state = self.__find_next_state(current_state, text[i])

            if self.out[current_state] == 0:
                continue
            for j in range(len(self.words)):
                if (self.out[current_state] & (1 << j)) > 0:
                    word = self.words[j]
                    result[word].append(i - len(word) + 1)

        return result
