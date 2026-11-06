import sys
from pathlib import Path
import re
import numpy as np
from collections import Counter



class word2Vec:

    # randomly initiliaze embeddings
    def initialize_embeddings(self):
        self.target_embeddings = np.random.uniform(
            -0.5 , 0.5, (len(self.V), self.D)
        )

        self.context_embeddings = np.random.uniform(
            -0.5 , 0.5, (len(self.V), self.D)
        )

    # generate pairs as positive and negative. its context window is 2
    def generate_training_pairs(self, tokens):
        pairs = []
        vocab_size = len(self.V)

        for target_position, target_word in enumerate(tokens):
            target_index = self.word_to_index[target_word]

            start = max(0, target_position - 2)
            end = min(len(tokens), target_position + 3)

            context_indices = [
                self.word_to_index[tokens[position]]
                for position in range(start, end)
                if position != target_position
            ]

            for context_index in context_indices:
                pairs.append((target_index, context_index, 1))

                negative_index = np.random.choice(
                    vocab_size,
                    p=self.negative_probabilities
                )

                pairs.append((target_index, negative_index, 0))

        return pairs

    def __init__(self):

        # Checking the number of args
        if len(sys.argv) != 3:
            print("Error: expected D and CORPUS_FILE.")
            raise SystemExit(1)

        # if given D is invalid, set it 50
        try:
            self.D = int(sys.argv[1])
        except ValueError:
            print("Invalid D. Using default D = 50.")
            self.D = 50

        if self.D <= 0:
            print("Invalid D. Using default D = 50.")
            self.D = 50

        # Checking the given file exsits in folder
        self.CORPUS_FILE = sys.argv[2]
        self.corpus_path = Path(__file__).parent / self.CORPUS_FILE

        if not self.corpus_path.is_file():
            print("Error: corpus file does not exist in the same folder.")
            raise SystemExit(1)

        # Creating Vocabulary from the given file by using white space tokenization
        with open(self.corpus_path, "r", encoding="utf-8") as file:
                text = file.read()

        text = re.sub(r"[^A-Za-z]", " ", text)
        text = text.lower()

        self.tokens = text.split()

        self.V = sorted(set(text.split()))

        #initalize word embeddings

        self.word_to_index = {
            word: index for index, word in enumerate(self.V)
        }

        self.initialize_embeddings()
        


    

    def main(self):


        word_counts = Counter(self.tokens)

        counts = np.array(
            [word_counts[word] for word in self.V],
            dtype=float
        )

        weights = counts ** 0.75
        self.negative_probabilities = weights / weights.sum()


        training_pairs = self.generate_training_pairs(self.tokens)
        print(training_pairs)





        


if __name__ == "__main__":
    word2Vec().main()