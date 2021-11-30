#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import re
#import math # May be needed for isnan
import html
import nltk
import spacy
import string
import unicodedata

global DEBUG
DEBUG = False

from bs4 import BeautifulSoup

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn 
from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.corpus import stopwords
stopword_list = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
#from nltk.tokenize.stanford import StanfordTokenizer

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

lemmatizer = WordNetLemmatizer()
tokenizer = ToktokTokenizer()

from IPython.display import display_markdown

def as_markdown(head='', body='Some body text'):
    if head is not '':
        display_markdown(f"#### {head}\n\n>{body}\n", raw=True)
    else:
        display_markdown(f">{body}\n", raw=True) 

# POS_TAGGER_FUNCTION : TYPE 1 
def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wn.ADJ 
    elif nltk_tag.startswith('V'): 
        return wn.VERB 
    elif nltk_tag.startswith('N'): 
        return wn.NOUN 
    elif nltk_tag.startswith('R'): 
        return wn.ADV 
    else:           
        return None

# Useful example code [here](https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/?ref=rp).
# 
# | Tag | What it Means |
# | :-- | :------------ |
# | CC | coordinating conjunction |
# | CD | cardinal digit |
# | DT | determiner |
# | EX | existential there (like: “there is” … think of it like “there exists”) |
# | FW | foreign word |
# | IN | preposition/subordinating conjunction |
# | JJ | adjective ‘big’ |
# | JJR | adjective, comparative ‘bigger’ |
# | JJS | adjective, superlative ‘biggest’ |
# | LS | list marker 1 |
# | MD | modal could, will |
# | NN | noun, singular ‘desk’ |
# | NNS | noun plural ‘desks’ |
# | NNP | proper noun, singular ‘Harrison’ |
# | NNPS | proper noun, plural ‘Americans’ |
# | PDT | predeterminer ‘all the kids’ |
# | POS | possessive ending parent‘s |
# | PRP | personal pronoun I, he, she |
# | PRP\$ | possessive pronoun my, his, hers |
# | RB | adverb very, silently, |
# | RBR | adverb, comparative better |
# | RBS | adverb, superlative best |
# | RP | particle give up |
# | TO | to go ‘to‘ the store. |
# | UH | interjection errrrrrrrm |
# | VB | verb, base form take |
# | VBD | verb, past tense took |
# | VBG | verb, gerund/present participle taking |
# | VBN | verb, past participle taken |
# | VBP | verb, sing. present, non-3d take |
# | VBZ | verb, 3rd person sing. present takes |
# | WDT | wh-determiner which |
# | WP  | wh-pronoun who, what |
# | WP\$ | possessive wh-pronoun whose |
# | WRB | wh-abverb where, when |

pos_tags = """| CC | coordinating conjunction |
| CD | cardinal digit |
| DT | determiner |
| EX | existential there |
| FW | foreign word |
| IN | preposition/subordinating conjunction |
| JJ | adjective ‘big’ |
| JJR | adjective, comparative |
| JJS | adjective, superlative |
| LS | list marker 1 |
| MD | modal could, will |
| NN | noun, singular |
| NNS | noun plural |
| NNP | proper noun, singular |
| NNPS | proper noun, plural |
| PDT | predeterminer |
| POS | possessive ending |
| PRP | personal pronoun |
| PRP | possessive pronoun |
| RB | adverb very, silently, |
| RBR | adverb, comparative |
| RBS | adverb, superlative |
| RP | participle |
| TO | to go ‘to‘ |
| UH | interjection |
| VB | verb, base form |
| VBD | verb, past tense |
| VBG | verb, gerund/present participle |
| VBN | verb, past participle |
| VBP | verb, sing. present, non-3d |
| VBZ | verb, 3rd person sing. present |
| WDT | wh-determiner |
| WP  | wh-pronoun |
| WP | possessive |
| WRB | wh-abverb |"""

global pos_lkp
pos_lkp  = {}
for p in pos_tags.split("\n"):
    m = re.search(r"^\| ([A-Z\$]{2,5})\s+\| ([^\|]+)", p)
    pos_lkp[m.groups()[0]] = m.groups()[1].strip()

def lemmatise_text(txt:str) -> list: # lemmatise tokens 
    tokenized = sent_tokenize(txt)
    processed = []
    for i in tokenized: 
      
        # Word tokenizers is used to find the words  
        # and punctuation in a string 
        wordsList = nltk.word_tokenize(i)
  
        # Using a part-of-speech  
        # tagger or POS-tagger.  
        tagged = nltk.pos_tag(wordsList) 
    
        for t in tagged:
            try:
                processed.append( (lemmatizer.lemmatize(t[0], pos=pos_tagger(t[1])), t[1]) )
            except KeyError:
                if DEBUG: print(f"Can't process: {t[0]} / {pos_tagger(t[1])}")
                processed.append( (t[0],None) )
    
    if DEBUG: print(f"Text with PoS tags: {processed}")
    return ' '.join([x[0] for x in processed])

# These seem to come through surprisingly often and need to resolved
def remove_quotemarks(text:str) -> str:
    smart_quotes = set(['‘','“','”','"','"',"’"])
    sq = re.compile(f"({'|'.join(smart_quotes)})",flags=re.IGNORECASE|re.DOTALL)
    text = re.sub("\w’\w","'",text,flags=re.IGNORECASE|re.DOTALL) # This is for possessives
    text = sq.sub(" ",text) # Get rid of everything else
    return text

# Adapted from https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html
def remove_accented_chars(text:str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

# From https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html
# From https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
global CONTRACTION_MAP
CONTRACTION_MAP = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "hhe is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "tthey would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "hat will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Adapted from https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html
def remove_special_chars(text:str, remove_digits:bool=False, replace_with_spaces:bool=True) -> str:
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    return re.sub(pattern, ' ' if replace_with_spaces else '', text)

# Adapted from https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html
global NUMBER_MAP
NUMBER_MAP = {
    'k': 1e3,
    'm': 1e6,
    'b': 1e9,
    'bn': 1e9,
    't': 1e12,
    'tn': 1e12
}

def expand_numbers(text:str, number_mapping=NUMBER_MAP):
    
    number_pattern = re.compile('([0-9]+([0-9.]*)?)({})\b'.format('|'.join(number_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_number_match(number):
        num    = float(number.group(1))
        suffix = number.group(len(number.groups()))
        
        exp = number_mapping.get(suffix)  if number_mapping.get(suffix)  else number_mapping.get(suffix.lower())
        return str(int(num * exp))
    
    expanded_txt = number_pattern.sub(expand_number_match, text)
    return expanded_txt

def strip_html_tags(doc:str) -> str:
    soup = BeautifulSoup(doc, "html.parser")
    return soup.get_text()

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#punkts = set([',',';',':','-','–','—','!','?']) # Doesn't work because of special chars
def remove_punctuation(text:str, keep_sentences:bool=True):
    pk = re.compile(r'[,;:\-!–?—\.\/\\\&]' if keep_sentences is True else r'[,;:\-!–?—]',flags=re.IGNORECASE|re.DOTALL)
    return pk.sub(" ",text) # Remove punctuation -- not needed if you remove special chars instead

def remove_short_text(doc:str, shortest_word:int=1):
    text = re.split('\s+',doc)
    punkts = re.compile(r'[,;:\-!–?—\.\/\\]',flags=re.IGNORECASE|re.DOTALL)
    return ' '.join([x for x in text if len(x)>shortest_word or punkts.match(x)])

def normalise_document(doc, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=False, 
                     punctuation_removal=True, keep_sentences=True, 
                     stopword_removal=True, remove_digits=False, infer_numbers=True, 
                     shortest_word=2):
    
    if DEBUG: print(f"Input:\n\t{doc}")

    try: 
        # strip HTML
        if html_stripping:
            # bs4 strips out semantically important whitespace so we need
            # to insert an extra space after end-tags.
            doc = re.sub(r'(\/[A-Za-z]+\d?|[A-Za-z]+ \/)>','\\1> ', html.unescape(doc))
            doc = strip_html_tags(doc)
            if DEBUG: print(f"After HTML removal:\n\t{doc}")
    
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
    
        # remove extra whitespace
        doc = re.sub('\s+', ' ', doc)
        if DEBUG: print(f"After newline and whitespace removal:\n\t{doc}")
        
        # remove quotemarks
        doc = remove_quotemarks(doc)
        
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            if DEBUG: print(f"After accent removal:\n\t{doc}")
    
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            if DEBUG: print(f"After contraction expansion:\n\t{doc}")
    
        # infer numbers from abbreviations
        if infer_numbers:
            doc = expand_numbers(doc)
            if DEBUG: print(f"After number expansion:\n\t{doc}")
    
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
            if DEBUG: print(f"After lower-casing:\n\t{doc}")
    
        # lemmatize text
        if text_lemmatization:
            doc = lemmatise_text(doc)
            if DEBUG: print(f"After lemmatisation:\n\t{doc}")
    
        # remove special characters and\or digits    
        if special_char_removal:
            doc = remove_special_chars(doc, remove_digits)
            if DEBUG: print(f"After special char removal:\n\t{doc}")
    
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            if DEBUG: print(f"After stopword removal:\n\t{doc}")
        
        # Deal with HTML entities -- not sure
        # why these aren't picked up earlier in 
        # the HTML function...
        doc = html.unescape(doc)
        
        # remove punctuation
        if punctuation_removal:
            doc = remove_punctuation(doc, keep_sentences)
            if DEBUG: print(f"After punctuation removal:\n\t{doc}")
        
        # remove short words
        if shortest_word > 1:
            doc = remove_short_text(doc, shortest_word)
            if DEBUG: print(f"After short text removal:\n\t{doc}")
        
        return doc
    except TypeError as err:
        if DEBUG:
            print(f"Problems with: {doc}")
            print(err)
            #traceback.print_exc(file=sys.stdout)
        rval = doc if doc is not None else ''
        return rval
