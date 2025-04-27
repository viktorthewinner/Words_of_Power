import json
from sentence_transformers import SentenceTransformer, util
import time

model_path = '/root/Github/bestem/model_trained' 
model = SentenceTransformer.load(model_path) 

start_time = time.time()

with open('words.json', 'r') as f:
    words_dict = json.load(f)

words_list = list(words_dict.keys())
input_word = "candle"

embeddings = model.encode([input_word] + words_list)
similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]  

sim_words = similarities.clone()

for i, word in enumerate(words_list):
    if word.lower() == input_word.lower():
        sim_words[i] = -1.0 


best_idx = sim_words.argmax()

best_word = words_list[best_idx]
best_cost = words_dict[best_word]
best_index = best_idx + 1

print(f"Cuvântul cel mai similar cu '{input_word}' este '{best_word}'")
print(f"Index în listă: {best_index}")
print(f"Cost asociat: {best_cost}")
print(f"Scor similaritate: {similarities[best_idx]:.2f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Timpul de rulare: {elapsed_time:.4f} secunde")
