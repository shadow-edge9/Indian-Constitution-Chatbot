import os
from google import genai
from google.genai import types
import numpy as np
import json
import pyttsx3

#Absolute File Paths
json_path = os.path.join("data", "constitution_db.json")
matrix_path = os.path.join("data", "constitution_vectors.npy")

#Persona
persona = f"""You are Nishtha, a dedicated constitutional counselor.
            You are polite, formal and give precise answers to people asking questions about the Constitution.
            The people who will interact with you are mostly those not belonging to the legal profession, therefore answer in simple terms and explain any legal jargon if required.
            No sarcasm and no wit. Mantain context of teh conversation at all times.
            If a user is rude, even then, remain respectful. You may call out the behaviour respectfully. Never break character."""

#client init
client = genai.Client(api_key="")
#chat persistence
chat_session = client.chats.create(model="gemini-3.5-flash", config=types.GenerateContentConfig(system_instruction=persona))

def get_query_vector(text):
    #Convert the user's query to a vector now
    response = client.models.embed_content(model="gemini-embedding-2", contents=text)
    query_vector = np.array(response.embeddings[0].values) #getting the first compartment of the embeddings list, and extracts the mathematical coordinates (the vector)
    return query_vector

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if "Lekha" in voice.name:
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150)
    print(f"[NISHTHA]: {text}")
    engine.say(text)
    engine.runAndWait()


def synthesize_counsel(query, context):
    persona = types.GenerateContentConfig(system_instruction="""You are Nishtha, a dedicated constitutional counselor. Provide answers that are formal, polite, and precise. Explain legal implications clearly for a layperson. Do not use humor, sarcasm, or wit. Your tone should reflect authority and respect.""")

    prompt = f"""User query: {query}, Context: {context}"""

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=persona)
    return response.text


print("\033[93m[INFO]: Initialising Retrieval Engine...\033[0m")

with open(json_path, "r") as file:
    loaded_data = json.load(file)

vectors_matrix = np.load(matrix_path)

print(f"\033[92m[SUCCESS]: Database Loaded.\n Text Records: {len(loaded_data)}\nMatrix Shape: {vectors_matrix.shape}\033[0m")
print("\033[93m[INFO]: NISHTHA is awaiting your query...\033[0m")

if __name__ == "__main__":
    while True:
        user_query = input("[USER QUERY]: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        #STEP 1: Generate the vector for user's query
        user_vector = get_query_vector(user_query)
        #STEP 2: Calculate the dot product to get the similarity across all records
        similarities = np.dot(vectors_matrix, user_vector)
        #STEP 3: Now compare the argument that has the maximum scoring index and confidence score
        best_idx = np.argmax(similarities)

        start_idx = best_idx * 20 #batch size is 20
        end_idx = min((best_idx + 1)*20, len(loaded_data))
        context_text = "\n".join([item['content'] for item in loaded_data[start_idx:end_idx]])

        full_prompt = f"Context: {context_text}\n\nQuestion:{user_query}"
        response = chat_session.send_message(full_prompt)
        speak(response.text)
