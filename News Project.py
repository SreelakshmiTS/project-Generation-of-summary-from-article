import streamlit as st

from bs4 import BeautifulSoup
import requests
import re
from collections import Counter 
from string import punctuation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
import pandas as pd
import pprint
from newsdataapi import NewsDataApiClient
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from heapq import nlargest

def tokenizer(s):
    tokens = []
    for word in s.split(' '):
        tokens.append(word.strip().lower())
        
    return tokens

def sent_tokenizer(s):
    sents = []
    for sent in s.split('.'):
        sents.append(sent.strip())
        
    return sents

def count_words(tokens):
    word_counts = {}
    for token in tokens:
        if token not in stop_words and token not in punctuation:
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1
                
    return word_counts

def word_freq_distribution(word_counts):
    freq_dist = {}
    max_freq = max(word_counts.values())
    for word in word_counts.keys():  
        freq_dist[word] = (word_counts[word]/max_freq)
        
    return freq_dist

def score_sentences(sents, freq_dist, max_len=40):
    sent_scores = {}  
    for sent in sents:
        words = sent.split(' ')
        for word in words:
            if word.lower() in freq_dist.keys():
                if len(words) < max_len:
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = freq_dist[word.lower()]
                    else:
                        sent_scores[sent] += freq_dist[word.lower()]
                        
    return sent_scores

def summarize(sent_scores, k):
    top_sents = Counter(sent_scores) 
    summary = ''
    scores = []
    
    top = top_sents.most_common(k)
    
    for t in top: 
        summary += t[0].strip() + '. '
        scores.append((t[1], t[0]))
        
    return summary[:-1], scores

def summarize_paragraph(paragraph, num_sentences):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    # Tokenize the sentences into words
    words = [word.lower() for sentence in sentences for word in nltk.word_tokenize(sentence)]
    # Remove stop words from the words list
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    # Calculate the word frequencies
    word_freq = nltk.FreqDist(filtered_words)
    # Calculate the sentence scores based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_freq:
                if i in sentence_scores:
                    sentence_scores[i] += word_freq[word]
                else:
                    sentence_scores[i] = word_freq[word]
    # Select the top N sentences with the highest scores
    summary_sentences_idx = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences_idx.sort()
    # Build the summary from the selected sentences
    summary = ' '.join([sentences[i] for i in summary_sentences_idx])
    original_text = summary
    if original_text:
        result_list = frequency_main(original_text, num_sentences)
    return summary

def frequency_main(text, sent_no):
    """
    Return a list of n sentences which is the summary of the text
    """

    result = []
    sents = sent_tokenize(text)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    freq = compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i, sent in enumerate(word_sent):
        for w in sent:
            if w in freq:
                ranking[i] += freq[w]
    sents_idx = rank(ranking, sent_no)
    [result.append(sents[j]) for j in sents_idx]
    return result

def compute_frequencies(word_sent):
    """
    Compute the frequency of each word.
    Input: word_sent, a list of sentences already tokenized.
    Output:freq, a dictionary where freq[w] is the frequency of w.
    """
    freq = defaultdict(int)
    for s in word_sent:
        for word in s:
            if word not in stopwords:
                freq[word] += 1
    # frequencies normalization and filtering

    m = max(freq.values())
    for w in list(freq):
        freq[w] = freq[w] / m
        if freq[w] >= max_cut or freq[w] <= min_cut:
            del freq[w]
    return freq

def paraphrase_sentence(sentence):
    # Load the GPT-2 model for text generation
    model = pipeline('text-generation', model='gpt2')
    words = word_tokenize(sentence)
    paraphrased_words = []
    for word in words:
        # Get the synonyms of the word
        synonyms = wordnet.synsets(word)
        if synonyms:
            # Choose a random synonym and add it to the paraphrased sentence
            paraphrased_word = synonyms[0].lemmas()[0].name()
            paraphrased_words.append(paraphrased_word)
        else:
            paraphrased_words.append(word)
    paraphrased_sentence = ' '.join(paraphrased_words)
    paraphrases = model(input_text, max_length=250, num_return_sequences=3)
    for i, paraphrase in enumerate(paraphrased_sentence):
        return paraphrase['generated_text']

st.title('Absrtractive Text Summarization')
st.subheader('A simple domain text summarizer made from scratch')

st.sidebar.subheader('Working of the project')

st.sidebar.markdown("* Build a solution around generation of short summary and appropriate abstraction based.")
st.sidebar.markdown("* The web application contains articles from the specific domains.")
st.sidebar.markdown("*  If the user search for a topic from the domain, it will create a short summary of the topic using abstractive summarization technique.")
st.sidebar.markdown("* Next, assign a score to the sentences by using the frequency distribution generated. This is simply summing up the scores of each word in a sentence. This function takes a max_len argument which sets a maximum length to sentences which are to be considered for use in the summarization. ")
st.sidebar.markdown("* In the final step, based on the scores, select the top 'k' sentences that represent the summary of the article. ")
st.sidebar.markdown("* Display the summary along with the top 'k' sentences and their sentence scores.")

url = st.text_input('\nEnter the topic')

no_of_sentences = st.number_input('Choose the no. of sentences in the summary', min_value = 1)
categories = st.selectbox(
    'Please select an option?',
    ('all','business', 'entertainment', 'environment', 'food', 'health','politics','science','sports','technology','tourism'),index=0)
    # ('business', 'entertainment', 'environment', 'food', 'health','politics','science','sports','technology','tourism'),index=0)
api = NewsDataApiClient(apikey="pub_1993022421979fb77d33eb743203ea4bfdcb3")
# response = api.news_api( q= url ,language = 'en', category = categories)
if categories == 'all':
    response = api.news_api( q= url ,language = 'en')
else:
    response = api.news_api( q= url ,language = 'en', category = categories)
if url and no_of_sentences and st.button('Summarize'):
    text = ""
    for i in range(0, len(response)):
        summary, summary_sent_scores = 0,0
        # print (len(response))
        if i > 5:
            break
        try:
            if  response["results"][i]['content']:
                print (i)
        except IndexError:
            continue
        else:
            text = response["results"][i]['content']
            if not text:
                print (text)
                continue
            text = re.sub(r'\[[0-9]*\]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            st.subheader('Original text: ')
            st.write(text)
            
            tokens = tokenizer(text)
            sents = sent_tokenizer(text)
            word_counts = count_words(tokens)
            freq_dist = word_freq_distribution(word_counts)
            sent_scores = score_sentences(sents, freq_dist)
            summary = summarize(sent_scores, no_of_sentences)
            summary, summary_sent_scores = summarize(sent_scores, no_of_sentences)

            
        
        
        
            st.subheader('Summarised text: ')
            
            summary = summarize_paragraph(text,no_of_sentences)
            summary = paraphrase_sentence(summary)
            st.write(summary)
        
            subh = 'Summary sentence score for the top ' + str(no_of_sentences) + ' sentences: '

            st.subheader(subh)
        
            data = []

            for score in summary_sent_scores: 
                data.append([score[1], score[0]])
                
            df = pd.DataFrame(data, columns = ['Sentence', 'Score'])

            st.table(df)
   