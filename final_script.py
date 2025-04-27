import requests
from time import sleep
from sentence_transformers import SentenceTransformer, util
import json
import time

host = "http://192.168.200.127:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 10

player_id = "avwsySin"

# -----------------------------------------------------------------------

print("Loading model and words...")
model_path = '/root/Github/bestem/model_trained'
model = SentenceTransformer.load(model_path)

with open('words.json', 'r') as f:
    words_dict = json.load(f)

words_list = list(words_dict.keys())
words_embeddings = model.encode(words_list)  
print("Model and words loaded.")

def register(player_id):
    register_url = f"{host}/register"
    data = {"player_id": player_id}
    response = requests.post(register_url, json=data)
    
    return response.json()
    
register(player_id)

# -----------------------------------------------------------------------

def what_beats(word):
    start_time = time.time()

    input_embedding = model.encode([word])[0]  
    similarities = util.cos_sim(input_embedding, words_embeddings)[0]

    sim_words = similarities.clone()

    for i, candidate_word in enumerate(words_list):
        if candidate_word.lower() == word.lower():
            sim_words[i] = -1.0  # penality

    best_idx = sim_words.argmax()

    best_word = words_list[best_idx]
    best_cost = words_dict[best_word]
    best_index = best_idx + 1 

    print(f"Cuvântul ales pentru '{word}' este '{best_word}' (index {best_index})")
    print(f"Cost asociat: {best_cost}")
    print(f"Scor similaritate: {similarities[best_idx]:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Timp de execuție: {elapsed_time:.4f} secunde\n")

    return int(best_index)

def play_game(player_id):

    def get_round():
        response = requests.get(get_url)
        print(response.json())
        sys_word = response.json()['word']
        round_num = response.json()['round']
        return (sys_word, round_num)

    submitted_rounds = []
    round_num = 0

    while round_num != NUM_ROUNDS :
        print(submitted_rounds)
        sys_word, round_num = get_round()
        while round_num == 0 or round_num in submitted_rounds:
            sys_word, round_num = get_round()
            sleep(0.5)

        if round_num > 1:
            status = requests.post(status_url, json={"player_id": player_id}, timeout=2)
            print(status.json())

        choosen_word = what_beats(sys_word)
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_num}
        response = requests.post(post_url, json=data, timeout=5)
        submitted_rounds.append(round_num)
        print("POST: !!!!!!!!!!!!!!!!")
        print(response.json())

# -----------------------------------------------------------------------

play_game(player_id)
