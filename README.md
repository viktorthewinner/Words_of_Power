Words of Power — Hackathon Project (partner: my sister)

The game is simple. The system sends a word through API request and I have to choose my own word (from the 77 words given) to "beat" the system.
Example: for system`s word "fire" I can choose "fire extinguisher" or "water".
Each word has a cost. If I don`t give a good answer, my bill is penalised with 75$.
The goal is to have the lowest bill possible.

Our solution: 
    using small LLM "all-MiniLM-L6-v2" for the similarity score
    training this model with the best 40 words for similarity (20 words to get beaten and 20 to beat)
    all the weights are changing, even for words we didn`t choose
    we return the word with the highest similarity between the system`s word and our list of words

Final statement about this 10 hours project:
    50% accuracy
    for better accuracy we could add a lot more training data (mentor from Veridion told us to have best accuracy we had to put around 8k words; we had 3.3k)
    tested my knowledge about LLMs
    achieved the experience of a hackathon
