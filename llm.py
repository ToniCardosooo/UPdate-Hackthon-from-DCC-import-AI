from openai import OpenAI
from scipy import spatial
import os

CLIENT = OpenAI()
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

TEXTS_EMBEDDED = []
EMBEDDINGS = []

AGENT_CONTEXT = """
    You are an agent that should produce a review about a specific game that will be given to you.
    You will receive a prompt formatted to a specific template with informations about the name of the game, the summary of the game and a predicted rating of the game in a scale of 0 to 100, each placed between '<' and '>'.
    The prompt will follow this template: <name of the game>,<summary of the game>,<predicted rating of the game>
    The review you create must be composed of three paragraphs.
    The first paragraph must contain the main feaures of the game, similar to a paraphrasing of the game summary.
    The second paragraph must have an overview of the possible positive points the game might have.
    The third and final paragraph must have an overview of the possible negative points the game might have.
    Everything you write in the second and last paragraphs must not be accurate descriptions but rather write that it might be like that description.
    You must follow this structure of answer always.
"""


def get_embedding(text, emb_model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return CLIENT.embeddings.create(input = [text], model=emb_model).data[0].embedding

def set_embeddings(df, emb_model="text-embedding-3-small"):
    global TEXTS_EMBEDDED, EMBEDDINGS
    if "emb.pkl" not in os.listdir():
        TEXTS_EMBEDDED = []
        EMBEDDINGS = []
        for i in range(df.shape[0]):
            text = f"{df.loc[i,'name']} | {df.loc[i,'genres']} | {df.loc[i,'summary']}"
            TEXTS_EMBEDDED.append(text)
            EMBEDDINGS.append(get_embedding(text, emb_model))
        with open("emb.pkl", "wb") as emb:
            pickle.dump(EMBEDDINGS, emb)
            emb.close()
        with open("text_emb.pkl", "wb") as text_emb:
            pickle.dump(TEXTS_EMBEDDED, text_emb)
            text_emb.close()
    else:
        with open("emb.pkl", "rb") as emb:
            EMBEDDINGS = pickle.load(emb)
            emb.close()
        with open("text_emb.pkl", "rb") as text_emb:
            TEXTS_EMBEDDED = pickle.load(text_emb)
            text_emb.close()


def get_prompt_context_strings(game_name, game_summary, predicted_rating, emb_model="text-embedding-3-small", relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n=5):
    user_prompt = f"{game_name} | {game_summary} | {predicted_rating}"
    query_embedding_response = CLIENT.embeddings.create(
        model=emb_model,
        input=user_prompt
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [(txt, relatedness_fn(query_embedding, EMBEDDINGS[i])) for i, txt in enumerate(TEXTS_EMBEDDED)]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def generate_context(strings):
    context_str = "Here are some names of similar and very similar games, their genres and their summaries that you can base on to build your own probable review."
    for s in strings:
        context_str += f"\n\n{s}"
    context_str += "\n\nPlease remember that everything you write in the second and last paragraphs must not be accurate descriptions but rather write that it might be like that description."
    return context_str

def generate_review(game_name, game_summary, predicted_rating, gpt_model="gpt-3.5-turbo", emb_model="text-embedding-3-small"):
    context_strings = get_prompt_context_strings(game_name, game_summary, predicted_rating, emb_model=emb_model)
    llm_output = CLIENT.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": f"{AGENT_CONTEXT}"},
            {"role": "system", "content": f"{generate_context(context_strings)}"},
            {"role": "user", "content": f"<{game_name}>,<{game_summary}>,<{predicted_rating}>"}
        ]
    )
    print(llm_output.choices[0].message.content)
    return llm_output.choices[0].message.content

import pandas as pd
import pickle

if __name__ == "__main__":
    df = pd.read_csv("toni.csv")
    df_emb = df.head(1000)
    set_embeddings(df_emb)
    generate_review("Grand Theft Auto 5", "Action game with heists", "90")
