
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


EMBEDDING_FILE = "CS585_P02A_A20656853_EMBED_50.TXT"
DEFAULT_K = 5


def load_embeddings(embedding_path):
    embeddings = {}

    with embedding_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if not row:
                continue

            word = row[0].strip().lower()

            vector = np.array(
                [float(value.strip()) for value in row[1:]],
                dtype=float
            )

            embeddings[word] = vector

    return embeddings


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


def find_nearest_neighbors(word, embeddings, k):
    target_vector = embeddings[word]
    similarities = []

    for other_word, vector in embeddings.items():
        if other_word == word:
            continue

        similarity = cosine_similarity(target_vector, vector)
        similarities.append((other_word, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


def plot_embeddings(word, neighbors, embeddings):
    # Target word + K nearest neighbors
    words = [word] + [neighbor for neighbor, _ in neighbors]

    vectors = np.array([
        embeddings[w] for w in words
    ])

    original_dimensions = vectors.shape[1]

    # Reduce dimensions to 3 using PCA
    if original_dimensions > 3:
        pca = PCA(n_components=3)
        coordinates = pca.fit_transform(vectors)
        method = "PCA"

    elif original_dimensions == 3:
        coordinates = vectors
        method = "None (already 3D)"

    else:
        # For D = 1 or D = 2, pad missing coordinates with zeros
        coordinates = np.zeros((len(words), 3))
        coordinates[:, :original_dimensions] = vectors
        method = "Zero padding"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot target word
    ax.scatter(
        coordinates[0, 0],
        coordinates[0, 1],
        coordinates[0, 2],
        c="red",
        s=120,
        label="Target word"
    )

    # Plot nearest neighbors
    ax.scatter(
        coordinates[1:, 0],
        coordinates[1:, 1],
        coordinates[1:, 2],
        c="blue",
        s=60,
        label="Nearest neighbors"
    )

    # Add word labels
    for i, w in enumerate(words):
        ax.text(
            coordinates[i, 0],
            coordinates[i, 1],
            coordinates[i, 2],
            w,
            fontsize=10
        )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")

    ax.set_title(
        f"Word2Vec: '{word}' and {len(neighbors)} Nearest Neighbors"
    )

    ax.legend()

    plt.tight_layout()
    plt.show()

    return method


def main():
    # Check number of arguments
    if len(sys.argv) != 3:
        print("Error: expected WORD and K.")
        raise SystemExit(1)

    # Check embedding file exists
    embedding_path = Path(__file__).parent / EMBEDDING_FILE

    if not embedding_path.is_file():
        print(
            f"Error: embedding file '{EMBEDDING_FILE}' "
            "does not exist in the same folder."
        )
        raise SystemExit(1)

    # Load embeddings
    embeddings = load_embeddings(embedding_path)
    vocabulary = set(embeddings.keys())

    # Validate WORD
    word = sys.argv[1]

    if not word or re.fullmatch(r"[A-Za-z]+", word) is None:
        print("Error: WORD must contain only Latin letters.")
        raise SystemExit(1)

    word = word.lower()

    if word not in vocabulary:
        print(f"Error: WORD '{word}' is not in the vocabulary.")
        raise SystemExit(1)

    # Validate K
    try:
        k = int(sys.argv[2])
    except ValueError:
        k = DEFAULT_K

    if not 1 <= k <= 10:
        k = DEFAULT_K

    # Check that enough neighbors exist
    if len(vocabulary) <= k:
        print(
            f"Error: vocabulary has only {len(vocabulary)} words. "
            f"Cannot find {k} distinct neighbors."
        )
        raise SystemExit(1)

    # Find K nearest neighbors in original D-dimensional space
    neighbors = find_nearest_neighbors(word, embeddings, k)

    # Determine dimensionality reduction method
    D = len(embeddings[word])

    if D > 3:
        method = "PCA"
    elif D == 3:
        method = "None (already 3D)"
    else:
        method = "Zero padding"

    # Required assignment output
    print("Serefoglu, Kivanc, A20656853 solution:")
    print(f"Word: {word}")
    print(f"Number of neighbors: {k}")
    print(f"Dimensionality reduction method: {method}")

    print("\nNearest neighbors:")
    for neighbor, similarity in neighbors:
        print(f"{neighbor}: {similarity:.6f}")

    # Visualize
    plot_embeddings(word, neighbors, embeddings)


if __name__ == "__main__":
    main()
