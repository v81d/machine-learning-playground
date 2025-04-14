import os
import spacy
import json
import asyncio
import time
from sklearn.ensemble import HistGradientBoostingClassifier


def init():
    os.system("clear")
    print(
        "\033[1;34m"
        + """                                                 .o.       ooooo                               
                                                .888.      `888'                               
 .oooo.        oooo d8b      oooo d8b          .8"888.      888       oooo    ooo       .oooo.o
`P  )88b       `888""8P      `888""8P         .8' `888.     888        `88.  .8'       d88(  "8
 .oP"888        888           888            .88ooo8888.    888         `88..8'        `"Y88b. 
d8(  888        888           888           .8'     `888.   888          `888'         o.  )88b
`Y888""8o      d888b         d888b         o88o     o8888o o888o          .8'          8""888P'
                                                                      .o..P'                   
                                                                      `Y8P'                    
"""
        + "\033[0m"
    )


async def loading_animation():
    global anim_start
    os.system("clear")
    anim_start = time.time()
    try:
        while True:
            for char in "|/-\\":
                print(
                    f'\rLoading base model "en_core_web_md" {char}', end="", flush=True
                )
                await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        pass


async def load_model_with_animation():
    global nlp
    animation_task = asyncio.create_task(loading_animation())
    nlp = await asyncio.to_thread(spacy.load, "en_core_web_md")
    animation_task.cancel()
    print(
        f'\rLoading base model "en_core_web_md" ... Done in {round(time.time() - anim_start, 2)}s!'
    )


asyncio.run(load_model_with_animation())

with open("./training/arrAIys/data.json", "r") as file:
    data = json.load(file)
    metadata = data["metadata"]
    training_data = data["data"]

unvectorized_input = list(training_data.keys())
labels = list(training_data.values())

vectors = [nlp(item.lower()).vector for item in unvectorized_input]

ai = HistGradientBoostingClassifier()
ai.fit(vectors, labels)  # vectors = input data, labels = output data

while True:
    # Introduction
    init()
    test = input("Welcome to arrAIys! Enter an input to test: ")
    vector = nlp(test).vector
    guess = "".join(map(str, ai.predict([vector]).tolist()))
    probas = ai.predict_proba([vector])

    # Results
    init()
    print(
        f'\033[1;32mClassification of [\033[1;36mtest: str = "{test}"\033[1;32m]: \033[1;33m{metadata[guess]}\033[1;32m (Confidence: \033[1;31m{max(probas[0]) * 100:.2f}%\033[1;32m)\033[0m'
    )

    resume = input("Do you want to test another input? (Y/n): \033[0m").lower()
    if resume == "n":
        break
