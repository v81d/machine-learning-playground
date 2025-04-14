import os
import spacy
import json
import asyncio
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)


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


async def loading_animation(message):
    global anim_start
    os.system("clear")
    try:
        while True:
            for char in "|/-\\":
                print(f"\r{message} {char}", end="", flush=True)
                await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        pass


async def run_with_animation(message, func, *args, **kwargs):
    animation_task = asyncio.create_task(loading_animation(message))
    result = await asyncio.to_thread(func, *args, **kwargs)
    animation_task.cancel()
    return result


async def load_model():
    global nlp
    nlp = await run_with_animation(
        'Loading base model "en_core_web_lg" ...', spacy.load, "en_core_web_lg"
    )


async def vectorize_data(items):
    return await run_with_animation(
        "Vectorizing training data ...",
        lambda x: [nlp(item.lower()).vector for item in x],
        items,
    )


async def train_model(model, X, y):
    return await run_with_animation(
        "Training the ensemble model ...", lambda m, x, y: m.fit(x, y), model, X, y
    )


async def main():
    await load_model()

    with open("./training/arrAIys/data.json", "r") as file:
        data = json.load(file)
        metadata = data["metadata"]
        training_data = data["data"]

    raw_input = list(training_data.keys())
    labels = list(training_data.values())

    vectors = await vectorize_data(raw_input)

    # Construct the ensemble model using a voting classifier
    # Amalgamates the predictions of multiple classifiers to reduce classifier bias and thus reinforce the final output
    ai = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42
                ),
            ),
            ("ab", AdaBoostClassifier(n_estimators=100, random_state=42)),
        ],
        voting="soft",  # 'soft' takes the average vote, while 'hard' takes the majority vote
        # Soft voting is better in this specific case, since the models return probabilities
    )
    # Note: In scikit-learn, an "estimator" is any object that can be trained on data and make predictions.

    await train_model(
        ai, vectors, labels
    )  # 'vectors' = input data, 'labels' = output data

    while True:
        # Front-end: prompt user for input
        init()
        user_input = input("Welcome to arrAIys! Enter an input to test: ")

        # Back-end: process input
        vector = nlp(user_input).vector
        guess = "".join(map(str, ai.predict([vector]).tolist()))
        probas = ai.predict_proba([vector])

        # Conclusion: display result
        init()
        print(
            f'\033[1;32mClassification of [\033[1;36muser_input: str = "\033[4;34m{user_input}\033[0m\033[1;36m"]\033[1;32m: \033[1;33m{metadata[guess]}\033[1;32m (Confidence: \033[1;31m{max(probas[0]) * 100:.2f}%\033[1;32m)\033[0m'
        )

        resume = input("\033[31marrAIys can make mistakes. Check important information.\033[0m Do you want to test another input? (Y/n): \033[0m").lower()
        if resume == "n":
            os.system("clear")
            break


asyncio.run(main())
