"""
 Use the sentence transformer to embed the input text using the gte-large-env1.5 model from Hugging Face.

 https://huggingface.co/sentence-transformers

 You can build an index of referents for each concept and save the index to a file using:

 from utils.embedder import Embedder
 embedder = Embedder()
 # for the first time, you can build the index for all concepts
 #embedder.build_index()
 embedder.build_index(["concept1", "concept2"], rebuild=True)

You can search the index for the most similar referents to a query using:
embedder.search("query", "concept_key", n=3)
"""
import os
import pickle
import time
from typing import Optional

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.loaders import load_concepts
from utils.subclasses import get_children

CACHE_FILE = "referent_embeddings.pkl"
MAX_BATCH_SIZE = 3000

class Embedder:
    def __init__(self, model:str="Alibaba-NLP/gte-large-en-v1.5", load_cache:bool=True):
        model_args = {}
        if torch.cuda.is_available():
            model_args["device"] = "cuda"
        elif torch.backends.mps.is_available():
            model_args["device"] = "mps"
        else:
            model_args["device"] = "cpu"

        # Note! the model will be downloaded to ~/.cache/huggingface/hub folder
        self.model = SentenceTransformer(model, trust_remote_code=True, **model_args)

        self.cache_path = os.path.join("cache", CACHE_FILE)
        self.cache = {}

        if load_cache:
            self.load()

    def embed(self, text: str) -> np.array:
        return self.model.encode(text)
    
    def embed_batch(self, texts: list[str], key:str|None=None) -> Optional[dict]:
        """
        Embed a batch of texts and store the embeddings in the cache if {key} is provided.
        """
        result = {"texts": [], "embeddings": []}

        if len(texts) > MAX_BATCH_SIZE:
            # split the batch into smaller batches
            n = len(texts)

            for i in range(0, n, MAX_BATCH_SIZE):
                batch_texts = texts[i:i+MAX_BATCH_SIZE]
                embeddings = self.model.encode(batch_texts)
                result["texts"].append(batch_texts)
                result["embeddings"].append(embeddings)
                print(f"Batch {i//MAX_BATCH_SIZE+1} of {n//MAX_BATCH_SIZE+1} done.")
        else:
            result["texts"] = texts
            result["embeddings"] = self.model.encode(texts)

        if key:
            self.cache[key] = result
            return
            
        return result

    def build_index(self, concepts:list[dict|str]|None=None, rebuild:bool=False):
        """
        Index the referents of each concept with embeddings and save the index.
        """
        if os.path.isfile(self.cache_path):
            print(f"Cache file '{self.cache_path}' already exists. Loading the index from cache.")
            self.load()
            if not rebuild:
                return

        if not concepts or all(isinstance(c, str) for c in concepts):
            concepts = load_concepts(concepts=concepts)
        for concept in concepts:
            referents = concept["referents"]
            if isinstance(referents, dict):
                referents = get_children(referents) 
            key = concept["concept"].lower()

            start = time.time()
            print(f"Going to index {len(referents)} referents for '{key}' ...", end=" ")
            self.embed_batch(referents, key)
            elapsed = time.time() - start
            print(f"done in {elapsed:.3f} seconds.")

        self.export()


    def export(self):
        """
        Export the cache to a file.
        """
        if self.cache:
            # make cache_path folder if it does not exist
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)

    def load(self):
        """
        Load the cache from a file.
        """
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
    
    def search(self, query:str, key:str, n:int=3) -> list[str]:
        """
        Search the cache under {key} and return {n} most similar texts
        based on increasing embedding distance from the {query}.
        """
        key = key.lower()
        if key not in self.cache:
            return []

        cache = self.cache[key]
        if not cache:
            return []

        embeddings = cache["embeddings"]
        query_embedding = self.embed(query)
        
        distances = []
        for batch_index in range(len(embeddings)):
            # use cosine distance to find the most similar embeddings
            distances.append(np.linalg.norm(embeddings[batch_index].reshape((-1, len(query_embedding))) - query_embedding, axis=1))

        distances = np.concatenate(distances)
        indices = np.argsort(distances)
        if len(cache["texts"]) == len(indices):
            # if the number of texts is equal to the number of indices, we can use the indices directly
            texts = cache["texts"]
        else:
            texts = [text for batch in cache["texts"] for text in batch]

        return [texts[i] for i in indices[:n]]
