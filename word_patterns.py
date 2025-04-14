import os; os.system("clear")  # CLEAR THE CONSOLE

import spacy
from sklearn.tree import DecisionTreeClassifier

nlp = spacy.load("en_core_web_md")  # WORD VECTORS/EMBEDDINGS FOR PATTERN RECOGNITION

# TRAINING DATA
unvectorized_input = [
    "stupid",
    "dumb",
    "idiot",
    "smart",
    "clever",
    "intelligent",
]
labels = [
    1,  # 1 = insult
    1,
    1,
    0,  # 0 = not insult
    0,
    0,
]

vectors = [nlp(item).vector for item in unvectorized_input]

# TRAIN THE AI WITH THE TRAINING DATA
ai = DecisionTreeClassifier()
ai.fit(vectors, labels)  # vectors = input data, labels = output data

user_input = input("Enter an input: ")  # TESTING INPUT!
vector = nlp(user_input).vector
guess = ai.predict([vector])

if guess[0] == 1:
    print(f"I think \"{user_input}\" is an INSULT.")
else:
    print(f"I think \"{user_input}\" is NOT an insult.")
