import re
import nltk
import processing.emoji as emoji
import sys
nltk.download('punkt')

twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
hashtag_re = re.compile(r'\B(\#[a-zA-Z0-9]+\b)(?!;)')
url_re = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
number_re = re.compile(r'[0-9]+')
lbr_re = re.compile(r'\|LBR\|')
emoji_re = re.compile(emoji._EMOJI_REGEXP)
retweet_1_re = re.compile(r'^RT ')
retweet_2_re = re.compile(r'^RT ')
html_re = re.compile(r'\&gt;')

def preprocess_text(text, tokenizer, stop_words):
    if not isinstance(text,str):
        return ''
    text = clean_text(text)
    text_tokens = tokenizer(text)
    text_tokens = [token.lemma_ for token in text_tokens]
    
    # remove stop words
    tokens_without_sw = [word.lower() for word in text_tokens if word.lower() not in stop_words]

    # merge to sentence
    cleaned_text = " ".join(tokens_without_sw)
    
    return cleaned_text

def clean_text(text):
    text = text.replace('\n', ' ')
    text = lbr_re.sub("", text)
    text = html_re.sub("", text)
    text = twitter_username_re.sub("",text)
    text = emoji_re.sub("",text)
    text = number_re.sub("",text)
    text = hashtag_re.sub("",text)
    text = url_re.sub("",text)
    text = retweet_1_re.sub("",text)
    text = retweet_2_re.sub("",text)
    

    #text = text.lower()
    return text

