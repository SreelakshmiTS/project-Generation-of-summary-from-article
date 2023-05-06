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
    return summary

def paraphrase_sentence(sentence):
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
    return paraphrased_sentence
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
api = NewsDataApiClient(apikey="pub_1993022421979fb77d33eb743203ea4bfdcb3")
response = api.news_api( q= url , country = "in")

if url and no_of_sentences and st.button('Summarize'):
    text = ""
    
    # r=requests.get(url)
    # soup = BeautifulSoup(r.content, 'html.parser') 
    # content = soup.find('div', attrs = {'id' : re.compile('content-body-14269002-*')})
    
    # for p in content.findChildren("p", recursive = 'False'):
    #     text+=p.text+" "
    # text = "In this article, we will explore Blockchain displaying increased access to the Metaverse and it is a boon for users This fantasy is becoming a reality thanks to blockchain technology , which also increases accessibility to the metaverse. The metaverse now has a vast array of opportunities thanks to blockchain. Not only that, but the Metaverse is also opening up new possibilities for developers, business owners, and entrepreneurs. Users now have more access to the Metaverse than ever before because of the rise of Blockchain-based platforms . The article mentions Blockchain’s power and how it has increased easy access to Metaverse. In the first place, it has improved security. Blockchain technology is also very secure, shielding user data and activities from nefarious individuals. Due to their dispersed design, decentralized networks are significantly more difficult to hack into than a single central database. Another big benefit of employing blockchain technologies is that they can drastically lower transaction costs. It also costs relatively little. Transactions are significantly more cost-effective because there are no expensive middlemen or third parties required. After all, all information is recorded on a distributed ledger. Several of the entry barriers in the metaverse have been reduced because of blockchain. Before the development of blockchain technology, getting access to the metaverse was a time-consuming and expensive procedure that demanded significant up-front payments in hardware and software licenses. Regardless of their degree of technical expertise or financial resources, blockchain has given developers access to the tools needed to make it simpler for people to join the metaverse. The way we engage with virtual worlds has been revolutionized by blockchain technology, which has made accessibility in the metaverse much more open and transparent. Platforms are now able to offer previously unheard-of levels of access to these new digital worlds by utilizing the power of blockchain, creating a level playing field where no one is left behind. Including enabling developers to produce and profit from their games and applications, to completely new types of governance in virtual economies. In conclusion, blockchain is enabling people to conduct business in previously unimaginable ways in virtual and augmented reality environments, ushering in a new era of accessibility to the metaverse. Blockchain technology is dismantling the barriers that have traditionally prevented users from taking part in and conducting business in the metaverse, from the enhanced security, trust, and control of blockchain-based infrastructure to the enhanced opportunities for both collaboration and monetization. Users now have greater and simpler access to the metaverse than ever before thanks to the power of blockchain. Disclaimer: The information provided in this article is solely the author/advertisers’ opinion and not an investment advice – it is provided for educational purposes only. By using this, you agree that the information does not constitute any investment or financial instructions by Analytics Insight and the team. Anyone wishing to invest should seek his or her own independent financial or professional advice. Do conduct your own research along with financial advisors before making any investment decisions. Analytics Insight and the team is not accountable for the investment views provided in the article."        
    # text = re.sub(r'\[[0-9]*\]', ' ', text)
    # text = re.sub(r'\s+', ' ', text)
    for i in range(0, 5):
        summary, summary_sent_scores = 0,0
        text = response["results"][i]['content']
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
        
        summary = summarize_paragraph(summary,no_of_sentences)
        # summary = paraphrase_sentence(summary)
        st.write(summary)
    
        subh = 'Summary sentence score for the top ' + str(no_of_sentences) + ' sentences: '

        st.subheader(subh)
    
        data = []

        for score in summary_sent_scores: 
            data.append([score[1], score[0]])
            
        df = pd.DataFrame(data, columns = ['Sentence', 'Score'])

        st.table(df)
   