#Meet Robo: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer # To represent sentences in a common format
from sklearn.metrics.pairwise import cosine_similarity # To find the similarity between sentences
import nltk
from nltk.stem import WordNetLemmatizer

# Reading the document
with open('chatbot.txt','r') as file:
    raw = file.read()

# converting the content of the document to lower case
raw = raw.lower()

# Tokenization: Split the document content into list of sentences
sent_tokens = nltk.sent_tokenize(raw)

# Lemmatization: For converting the words to their base form
lemmer = WordNetLemmatizer()

punctuation_mark_dictionary = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    text_lower = text.lower()
    text_without_punctuation = text_lower.translate(punctuation_mark_dictionary)
    words = nltk.word_tokenize(text_without_punctuation)
    lemmatized_words = [lemmer.lemmatize(word) for word in words]
    return lemmatized_words

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def is_greeting_message(sentence):
    """If user's input is a greeting, return a True"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return True
    
    return False

# Generating response
def find_response_from_document(user_response):
    robo_response=''
    sent_tokens.append(user_response)

    # create object of TF IDF vectorizer
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

    # convert document sentence tokens to TF IDF vectors
    all_tfidf_vectors = TfidfVec.fit_transform(sent_tokens)
 
    # compute similarity between user input and all rest document sentences
    vals = cosine_similarity(all_tfidf_vectors[-1], all_tfidf_vectors)
 
    # select the document sentence which has the highest similarity score
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    highest_score_similarity_sentence = flat[-2]

    # form the response

    if(highest_score_similarity_sentence==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        sent_tokens.remove(user_response)
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        sent_tokens.remove(user_response)
        return robo_response

def botrespond(user_response):

    user_response = user_response.lower()

    is_user_greeting = is_greeting_message(user_response)

    if(is_user_greeting == True):
        return random.choice(GREETING_RESPONSES)
    elif(user_response=='thanks' or user_response=='thank you' ):
        return("You are welcome..")
    elif(user_response == 'bye'):
        return("Bye! take care..")
    else:
        res = find_response_from_document(user_response)
        return res
