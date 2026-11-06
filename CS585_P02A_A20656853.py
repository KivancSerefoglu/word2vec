import sys
from pathlib import Path
import re
import numpy as np


class word2Vec:

    def initialize_embeddings(self):
        self.target_embeddings = np.random.uniform(
            -0.5 , 0.5, (len(self.V), self.D)
        )

        self.context_embeddings = np.random.uniform(
            -0.5 , 0.5, (len(self.V), self.D)
        )

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
        
        text = re.sub(r"[^A-Za-z\s]", "", text)
        text = text.lower()

        self.V = sorted(set(text.split()))

        #initalize word embeddings

        self.word_to_index = {
            word: index for index, word in enumerate(self.V)
        }

        self.initialize_embeddings()


        


    

    def main(self):
        print(self.target_embeddings[self.word_to_index["alice"]])





        


if __name__ == "__main__":
    word2Vec().main()