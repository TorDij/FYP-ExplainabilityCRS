import pandas as pd
import numpy as np
import json
import time
import subprocess
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderPromptEnhancer:
    def __init__(self):
        # read the original KG and store
        self.big_entities_path = r"/home/Nema/UniCRS_GraphRAG/GraphRAG/output/successful_20250129-110435/artifacts/create_final_entities.parquet"
        self.big_relationships_path = r"/home/Nema/UniCRS_GraphRAG/GraphRAG/output/successful_20250129-110435/artifacts/create_final_relationships.parquet"
        self.dialogue_entities_path = r"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output/create_final_entities.parquet"
        self.dialogue_relationships_path = r"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output/create_final_relationships.parquet"
        # Load a sentence embedding model
        self.relationship_model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_dialogue_kg(self, line):
        line = line.strip()
        with open('/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/input/current_line.txt', 'w') as current_file:
            current_file.write(line)

        # Run the graphrag command and wait for it to complete
        process = subprocess.run(['graphrag', 'index', '--root', '/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/'], 
                            capture_output=True,  
                            text=True) 
        
        if process.returncode == 0:
            print("Command completed successfully")

    def load_data(self, big_entities_path, big_relationships_path, dialogue_entities_path, dialogue_relationships_path):
        big_entities = pd.read_parquet(big_entities_path)
        big_relationships = pd.read_parquet(big_relationships_path)
        dialogue_entities = pd.read_parquet(dialogue_entities_path)
        dialogue_relationships = pd.read_parquet(dialogue_relationships_path)
        return big_entities, big_relationships, dialogue_entities, dialogue_relationships

    def generate_embeddings(self, texts, model_name='all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_tensor=True)
        return embeddings

    def find_similar_entities(self, big_entities, dialogue_entities, threshold=0.8, top_n=10):
        big_texts = big_entities['description'].fillna('')
        dialogue_texts = dialogue_entities['description'].fillna('')
        
        big_embeddings = self.generate_embeddings(big_texts.tolist())
        dialogue_embeddings = self.generate_embeddings(dialogue_texts.tolist())
        
        similarity_matrix = cosine_similarity(dialogue_embeddings.cpu(), big_embeddings.cpu())
        
        similar_entities = set()
        for i, row in enumerate(similarity_matrix):
            top_matches = np.argsort(row)[-top_n:][::-1]  
            for j in top_matches:
                if row[j] >= threshold:
                    similar_entities.add(big_entities.iloc[j]['title'])
        
        return similar_entities

    def extract_subgraph(self, big_entities, big_relationships, similar_entities):
        subgraph_entities = big_entities[big_entities['title'].isin(similar_entities)][['title', 'description', 'type']]
        subgraph_relationships = big_relationships[
            (big_relationships['source'].isin(similar_entities)) | (big_relationships['target'].isin(similar_entities))
        ]
        return subgraph_entities, subgraph_relationships

    # def save_combined_relationship_descriptions(self, subgraph_relationships, output_path="combined_relationship_descriptions.txt"):
    #     combined_descriptions = " ".join(subgraph_relationships["description"].dropna())
    #     with open(output_path, "w", encoding="utf-8") as file:
    #         file.write(combined_descriptions)

    def get_relationship_embeddings(self, subgraph_relationships):
        """Convert relationship descriptions into embeddings"""
        combined_descriptions = " ".join(subgraph_relationships["description"].dropna())

        # Generate embeddings (Current size likely 384)
        relationship_embeddings = torch.tensor(self.relationship_model.encode(combined_descriptions))

        # Ensure correct shape: (batch_size, seq_len, hidden_size)
        if relationship_embeddings.dim() == 1:  # If it's (hidden_size,), reshape
            relationship_embeddings = relationship_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)

        # Ensure hidden size is compatible (Expand from 384 to 768 if needed)
        required_hidden_size = 768  # Match model's expected size
        if relationship_embeddings.shape[-1] != required_hidden_size:
            relationship_embeddings = torch.nn.functional.pad(
                relationship_embeddings, 
                (0, required_hidden_size - relationship_embeddings.shape[-1])  # Pad to 768
            )

        # Move embeddings to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return relationship_embeddings.to(device)


    def get_enhanced_rec_prompt(self, line):
        print(f"\n[DEBUG] Processing Dialogue for Relationships: {line}\n")

        # Ensure KG is created
        self.create_dialogue_kg(line)
        
        big_entities, big_relationships, dialogue_entities, dialogue_relationships = self.load_data(
            self.big_entities_path, self.big_relationships_path, self.dialogue_entities_path, self.dialogue_relationships_path)

        similar_entities = self.find_similar_entities(big_entities, dialogue_entities, threshold=0.75, top_n=10)
        subgraph_entities, subgraph_relationships = self.extract_subgraph(big_entities, big_relationships, similar_entities)

        # If no relationships found, return None
        if subgraph_relationships.empty:
            print("[WARNING] No relationships found for this dialogue!")
            return None

        relationship_embeddings = self.get_relationship_embeddings(subgraph_relationships)

        # Debugging print
        print(f"[DEBUG] Extracted Relationship Embeddings Shape: {relationship_embeddings.shape}")
        
        return relationship_embeddings


        
        # subgraph_entities.to_csv("subgraph_entities.csv", index=False)
        # subgraph_relationships.to_csv("subgraph_relationships.csv", index=False)

        # self.save_combined_relationship_descriptions(subgraph_relationships)
        
        # print("Subgraph extraction complete. Check 'subgraph_entities.csv' and 'subgraph_relationships.csv'")



    