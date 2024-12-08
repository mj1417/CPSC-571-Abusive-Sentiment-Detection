import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Downloads a list of common English stopwords
nltk.download('stopwords')

# Downloads a pre-trained tokenizer
nltk.download('punkt')

# Downloads the WordNet lexical dataset
nltk.download('wordnet')

# Loading the testing and training dataset into DataFrames
df1 = pd.read_csv("testing dataset.csv")
df2 = pd.read_csv("training dataset.csv")

# Initialize stop words, lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Slang dictionary created by looking at the data and modern slangs in use today
slang_dict = {
"u": "you",
"ur": "your",
"gr8": "great",
"4u": "for you",
"plz": "please",
"smh": "shaking my head",
"lmao": "laughing my ass off",
"lol": "laugh out loud",
"wtf": "what the fuck",
"omg": "oh my god",
"idk": "I don't know",
"af": "as fuck",
"jk": "just kidding",
"brb": "be right back",
"fam": "family",
"bae": "significant other",
"lit": "exciting",
"dope": "cool",
"rn": "right now",
"yo": "hey",
"wya": "where you at",
"sus": "suspicious",
"dm": "direct message",
"noob": "newbie",
"thot": "that hoe over there",
"flex": "show off",
"ghost": "suddenly leave",
"hmu": "hit me up",
"salty": "bitter",
"lowkey": "not obvious",
"highkey": "very obvious",
"slay": "succeed",
"turnt": "excited",
"vibe": "feeling",
"clapback": "quick response",
"shade": "disrespect",
"stan": "obsessive fan",
"cap": "lie",
"no cap": "no lie",
"wyd": "what are you doing",
"fwb": "friends with benefits",
"defs": "definitely",
"yass": "yes",
"on fleek": "perfect",
"tbh": "to be honest",
"imo": "in my opinion",
"irl": "in real life",
"fomo": "fear of missing out",
"idc": "I don't care",
"ngl": "not gonna lie",
"otp": "on the phone",
"grind": "work hard",
"simp": "overly attentive person",
"thicc": "curvy",
"bussin": "really good",
"pog": "play of the game",
"fire": "amazing",
"basic": "unoriginal",
"extra": "dramatic",
"bop": "catchy song",
"dead": "very amused",
"woke": "socially aware",
"sus": "suspicious",
"bro": "brother",
"sis": "sister",
"cuz": "because",
"nah": "no",
"ya": "you",
"gonna": "going to",
"ain't": "is not",
"luv": "love",
"thx": "thanks",
"fr": "for real",
"w/e": "whatever",
"kinda": "kind of",
"ima": "I'm going to",
"lemme": "let me",
"dope": "awesome",
"turnt": "excited or energetic",
"np": "now playing",
"bday": "birthday",
"qtna": "questions that need answers",
"np": "no problem",
"irl": "in real life",
"yolo": "you only live once",
"slaps": "good music or highly approved",
"clout": "influence or fame",
"no cap": "no lie, being honest",
"fam": "family or close friend",
"extra": "dramatic, over the top",
"bop": "a catchy song",
"vibe": "mood or atmosphere",
"salty": "bitter, upset",
"spill the tea": "share gossip",
"goat": "greatest of all time",
"bet": "okay, sure",
"fire": "amazing, excellent",
"flex": "show off",
"lowkey": "not obvious, subtly",
"highkey": "obvious or apparent",
"stan": "obsessive fan",
"turnt": "excited, hyper, often due to partying",
"dope": "cool, awesome",
"grind": "work hard, hustle",
"fomo": "fear of missing out",
"savage": "ruthless, doing something with no regard",
"ghost": "to cut off all communication suddenly",
"snatched": "well put together, stylish",
"bougie": "acting upscale or high class",
"shade": "subtle insult",
"woke": "socially aware",
"lit": "exciting, fun",
"sus": "suspicious",
"simp": "overly attentive or submissive to someone they like"
}

# Function to expand slang and abbreviations
def expand_slang(text):
    words = text.split()
    return ' '.join([slang_dict.get(word.lower(), word) for word in words])

# Preprocessing function
def preprocess_text(text):

    # Check if text is not a string (e.g., NaN or other types)
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()
    # Remove special characters and only keeps letters and numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Expand slang and abbreviations
    text = expand_slang(text)
    # Tokenize
    words = re.findall(r'\b\w+\b', text)
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words back into a single string
    return ' '.join(words)

# Apply preprocessing to the 'comment' column
df1['processed_comment'] = df1['comment'].apply(preprocess_text)
df2['processed_comment'] = df2['comment'].apply(preprocess_text)

# Currently the class column is classified as
# 0 for hate speech, 1 for offensive language and 2 for neither
# we have changed the classification as
# 1 for abusive sentiment and 0 for nothing
df1['label'] = df1['class'].apply(lambda x: 1 if x in [0, 1] else 0)
df2['label'] = df2['class'].apply(lambda x: 1 if x in [0, 1] else 0)


# Save processed data to a new CSV
df1.to_csv("processed_testing_dataset.csv", index=False)
print("Successfully created the processed testing data file\n")

df2.to_csv("processed_training_dataset.csv", index=False)
print("Successfully created the processed training data file\n")

