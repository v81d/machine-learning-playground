import os; os.system("clear")  # CLEAR THE CONSOLE

import spacy
import json
from sklearn.tree import DecisionTreeClassifier

nlp = spacy.load("en_core_web_md")  # WORD VECTORS/EMBEDDINGS FOR PATTERN RECOGNITION

# LOAD TRAINING DATA FROM JSON FILE
with open("./training/arrAIys/data.json", "r") as file:
    training_data = json.load(file)

unvectorized_input = list(training_data.keys())
labels = list(training_data.values())

vectors = [nlp(item).vector for item in unvectorized_input]

# TRAIN THE AI WITH THE TRAINING DATA
ai = DecisionTreeClassifier()
ai.fit(vectors, labels)  # vectors = input data, labels = output data

print("""

                                                 .o.       ooooo                               
                                                .888.      `888'                               
 .oooo.        oooo d8b      oooo d8b          .8"888.      888       oooo    ooo       .oooo.o
`P  )88b       `888""8P      `888""8P         .8' `888.     888        `88.  .8'       d88(  "8
 .oP"888        888           888            .88ooo8888.    888         `88..8'        `"Y88b. 
d8(  888        888           888           .8'     `888.   888          `888'         o.  )88b
`Y888""8o      d888b         d888b         o88o     o8888o o888o          .8'          8""888P'
                                                                      .o..P'                   
                                                                      `Y8P'                    
""")

user_input = input("Welcome to arrAIys! Enter an input to test: ")
vector = nlp(user_input).vector
guess = ai.predict([vector])

if guess[0] == 1:
    print(f"I think \"{user_input}\" is a FOOD.")
else:
    print(f"I think \"{user_input}\" is NOT a food.")