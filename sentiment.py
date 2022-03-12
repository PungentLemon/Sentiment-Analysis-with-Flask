import re
import joblib
import nltk
from nltk.corpus import stopwords

# load the sentiment model
with open("sentiment_model_pipeline.pkl", "rb") as f:
    model = joblib.load(f)

def remove_spl_char(text):
    
    text = text.lower()
    
    text = re.sub('https?://\S+|www\.\S+', ' ', text) #it's used for remove urls
    text = re.sub('<.*?>', ' ', text) #it's used for remove html parses
    text = re.sub("[^\w\s\d]", " ", text) # it's used for remove punctuations
    text = re.sub("[^a-zA-Z0-9]+", " ", text)
    
    return text

#Removing of stopwords

def remove_stopwords_and_lemmatization(text):
    final_text=[]
    text= text.lower()
    text= nltk.word_tokenize(text)
    
    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word= lemma.lemmatize(word)
            final_text.append(word)
    return " ".join(final_text)

def clean_data(data):
    text= remove_spl_char(data)
    text= remove_stopwords_and_lemmatization(text)
    return text