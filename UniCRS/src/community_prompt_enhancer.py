import pandas as pd
import numpy as np
import json
import time
import torch
import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

class CommunityRecommenderPromptEnhancer:
    def __init__(self):
        # Paths to the original KG and dialogue KG community reports
        self.big_community_reports_path = r"/home/Nema/UniCRS_GraphRAG/GraphRAG/output/successful_20250129-110435/artifacts/create_final_community_reports.parquet"
        self.dialogue_community_reports_base_path = r"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output"
        
        # For embedding generation
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cache to store big community data and embeddings
        self._big_communities_cache = None
        self._big_embeddings_cache = None
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = self.embedding_model.to(self.device)
        
        # Batch size for processing
        self.batch_size = 32
        
        # Cache directory for processed dialogues
        self.cache_dir = "/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/embedding_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Preload big communities data (one-time operation)
        try:
            print("Preloading big communities data...")
            self._preload_big_communities()
            print("Preloading complete!")
        except Exception as e:
            print(f"Warning: Could not preload big communities: {e}")
    
    def _preload_big_communities(self):
        """Preload big communities data at initialization for better performance"""
        if self._big_communities_cache is None:
            big_community_path = self.big_community_reports_path
            start_time = time.time()
            
            # Read big communities CSV file (this happens only once)
            big_communities = pd.read_parquet(big_community_path)
            
            # Process JSON columns efficiently - only for essential columns
            essential_cols = ['community', 'title', 'summary', 'findings']
            cols_to_keep = [col for col in essential_cols if col in big_communities.columns]
            big_communities = big_communities[cols_to_keep].copy()
            
            for col in ['findings']:
                if col in big_communities.columns:
                    try:
                        # Convert JSON strings to objects
                        string_mask = big_communities[col].apply(
                            lambda x: isinstance(x, str) and x.strip().startswith('{')
                        )
                        if string_mask.any():
                            big_communities.loc[string_mask, col] = big_communities.loc[string_mask, col].apply(
                                lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith('{') else x
                            )
                    except Exception as e:
                        print(f"Warning: Error processing JSON in column {col}: {e}")
            
            # Cache the processed data
            self._big_communities_cache = big_communities
            print(f"Preloaded {len(big_communities)} big communities in {time.time() - start_time:.2f} seconds")

    def get_latest_dialogue_report(self, step):
        """Get the most recent dialogue community report for the given step"""
        report_pattern = os.path.join(self.dialogue_community_reports_base_path, f"train_step{step}", "artifacts/create_final_community_reports.parquet")
        print(report_pattern)
        
        try:
            if os.path.exists(report_pattern):
                print(f"Found dialogue report at: {report_pattern}")
                return report_pattern
            else:
                print(f"Warning: No dialogue report found for step {step}")
                return None
        except Exception as e:
            print(f"Error finding dialogue report: {e}")
            return None

    def load_community_data(self, big_community_path, dialogue_community_path):
        """Load community reports from both the big KG and dialogue KG with caching for big communities"""
        try:
            # Use cached big communities if available
            if self._big_communities_cache is not None:
                big_communities = self._big_communities_cache
                print("Using cached big communities data")
            else:
                # Read big communities CSV file (this happens only once)
                start_time = time.time()
                big_communities = pd.read_parquet(big_community_path)
                
                # Process JSON columns efficiently - only for essential columns
                essential_cols = ['community', 'title', 'summary', 'findings']
                big_communities = big_communities[essential_cols].copy()
                
                for col in ['findings']:
                    if col in big_communities.columns:
                        try:
                            # Use vectorized operations instead of apply when possible
                            mask = big_communities[col].astype(str).str.startswith('{')
                            if mask.any():
                                big_communities.loc[mask, col] = big_communities.loc[mask, col].apply(json.loads)
                        except Exception as e:
                            print(f"Warning: Could not parse JSON in column {col}: {e}")
                
                # Cache for future use
                self._big_communities_cache = big_communities
                print(f"Loaded big communities in {time.time() - start_time:.2f} seconds")
            
            # Read dialogue communities CSV file (this happens for each dialogue)
            dialogue_communities = pd.read_parquet(dialogue_community_path)
            
            # Process only essential columns for dialogue communities
            essential_cols = ['community', 'title', 'summary', 'findings']
            cols_to_keep = [col for col in essential_cols if col in dialogue_communities.columns]
            dialogue_communities = dialogue_communities[cols_to_keep].copy()
            
            for col in ['findings']:
                if col in dialogue_communities.columns:
                    try:
                        mask = dialogue_communities[col].astype(str).str.startswith('{')
                        if mask.any():
                            dialogue_communities.loc[mask, col] = dialogue_communities.loc[mask, col].apply(json.loads)
                    except Exception as e:
                        print(f"Warning: Could not parse JSON in column {col}: {e}")
            
            print(f"Loaded {len(dialogue_communities)} dialogue communities")
            
            return big_communities, dialogue_communities
            
        except Exception as e:
            print(f"Error loading community data: {e}")
            # Return empty DataFrames to gracefully handle the error
            return pd.DataFrame(), pd.DataFrame()

    def generate_embeddings(self, texts, batch=True):
        """Generate embeddings for a list of texts with batching for efficiency"""
        if not texts:
            return torch.tensor([])
            
        if batch and len(texts) > self.batch_size:
            # Process in batches for memory efficiency
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device
                )
                all_embeddings.append(batch_embeddings)
            return torch.cat(all_embeddings)
        else:
            # Process small lists without batching
            return self.embedding_model.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device
            )

    def find_similar_communities(self, big_communities, dialogue_communities, threshold=0.75, top_n=5):
        """Find communities in the big KG that are similar to the dialogue communities with optimization for speed"""
        # Fast string conversion function for vectors
        def fast_safe_str(series):
            # Vectorized operations where possible
            result = series.fillna("")
            # Handle dictionaries and lists
            mask = result.apply(lambda x: isinstance(x, (dict, list)))
            if mask.any():
                result.loc[mask] = result.loc[mask].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
            return result.astype(str)
        
        # Generate text representations more efficiently
        start_time = time.time()
        big_texts = fast_safe_str(big_communities['summary']) + ' ' + fast_safe_str(big_communities['findings'])
        dialogue_texts = fast_safe_str(dialogue_communities['summary']) + ' ' + fast_safe_str(dialogue_communities['findings'])
        
        # Generate/use cached embeddings for big communities
        if self._big_embeddings_cache is None:
            print("Generating embeddings for big communities (one-time operation)...")
            self._big_embeddings_cache = self.generate_embeddings(big_texts.tolist())
        big_embeddings = self._big_embeddings_cache
        
        # Generate embeddings for dialogue communities (much smaller)
        print("Generating embeddings for dialogue communities...")
        dialogue_embeddings = self.generate_embeddings(dialogue_texts.tolist())
        
        print(f"Embeddings generated in {time.time() - start_time:.2f} seconds")
        
        # Calculate similarity efficiently
        start_time = time.time()
        # Move tensors to CPU only once before calculation
        if torch.is_tensor(dialogue_embeddings):
            dialogue_embeddings_cpu = dialogue_embeddings.cpu().numpy()
        else:
            dialogue_embeddings_cpu = dialogue_embeddings
            
        if torch.is_tensor(big_embeddings):
            big_embeddings_cpu = big_embeddings.cpu().numpy()
        else:
            big_embeddings_cpu = big_embeddings
            
        similarity_matrix = cosine_similarity(dialogue_embeddings_cpu, big_embeddings_cpu)
        print(f"Similarity calculated in {time.time() - start_time:.2f} seconds")
        
        # Find top similar communities efficiently
        start_time = time.time()
        similar_communities = []
        
        # Pre-fetch community IDs to avoid repeated iloc operations
        dialogue_community_ids = dialogue_communities['community'].values
        big_community_ids = big_communities['community'].values
        
        for i, row in enumerate(similarity_matrix):
            # Get top indices directly with argpartition (faster than argsort for top-k)
            top_indices = np.argpartition(row, -top_n)[-top_n:]
            # Sort only the top subset
            top_indices = top_indices[np.argsort(-row[top_indices])]
            
            dialogue_community_id = dialogue_community_ids[i]
            
            for j in top_indices:
                similarity_score = row[j]
                if similarity_score >= threshold:
                    big_community_id = big_community_ids[j]
                    # Only fetch the necessary data from the dataframe
                    similar_communities.append({
                        'dialogue_community_id': int(dialogue_community_id),
                        'big_community_id': int(big_community_id),
                        'similarity_score': float(similarity_score),
                        'big_community_data': big_communities.iloc[j].to_dict()
                    })
        
        print(f"Found {len(similar_communities)} similar communities in {time.time() - start_time:.2f} seconds")
        return similar_communities

    def extract_community_content(self, big_communities, similar_communities):
        """Extract content from similar communities in the big KG"""
        all_relevant_content = []
        
        for match in similar_communities:
            big_community_id = match['big_community_id']
            community_data = match['big_community_data']
            
            # Safe extraction with default values
            def safe_get(key, default=""):
                value = community_data.get(key, default)
                # Handle array/Series type values
                if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                    return str(value[0]) if len(value) > 0 else default
                elif pd.isna(value) or value is None:
                    return default
                elif isinstance(value, (dict, list)):
                    return json.dumps(value)
                else:
                    return str(value)
            
            # Extract important information from the community
            content = {
                'community_id': big_community_id,
                'title': safe_get('title'),
                'summary': safe_get('summary'),
                'findings': safe_get('findings'),
                'similarity_score': match['similarity_score']
            }
            
            all_relevant_content.append(content)
        
        return all_relevant_content

    def get_community_embeddings(self, community_content):
        """Convert community content into embeddings suitable for recommendation"""
        # Combine all content into a single text
        combined_text = ""
        for content in community_content:
            combined_text += f"Community {content['community_id']}: {content['title']}. "
            combined_text += f"Summary: {content['summary']}. "
            combined_text += f"Findings: {content['findings']}. "
        
        # Generate embeddings
        community_embeddings = torch.tensor(self.embedding_model.encode(combined_text))
        
        # Ensure correct shape: (batch_size, seq_len, hidden_size)
        if community_embeddings.dim() == 1:  # If it's (hidden_size,), reshape
            community_embeddings = community_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)

        # Ensure hidden size is compatible (Expand from 384 to 768 if needed)
        required_hidden_size = 768  # Match model's expected size
        if community_embeddings.shape[-1] != required_hidden_size:
            community_embeddings = torch.nn.functional.pad(
                community_embeddings, 
                (0, required_hidden_size - community_embeddings.shape[-1])  # Pad to 768
            )

        # Move embeddings to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return community_embeddings.to(device)

    def get_dialogue_hash(self, step):
        """Generate a hash for the dialogue step for caching purposes"""
        import hashlib
        return hashlib.md5(f"step_{step}".encode()).hexdigest()
        
    def get_cached_embeddings(self, dialogue_hash):
        """Try to retrieve cached embeddings for a dialogue"""
        cache_path = os.path.join(self.cache_dir, f"{dialogue_hash}.pt")
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location=self.device)
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
        return None
        
    def save_embeddings_to_cache(self, dialogue_hash, embeddings):
        """Save embeddings to cache for future use"""
        cache_path = os.path.join(self.cache_dir, f"{dialogue_hash}.pt")
        try:
            torch.save(embeddings, cache_path)
        except Exception as e:
            print(f"Error saving embeddings to cache: {e}")
    
    def get_enhanced_rec_prompt(self, step, dialogue_text):
        """Main function to generate enhanced recommendation prompts based on community overlap"""
        start_total = time.time()
        print(f"\n[DEBUG] Processing step: {step}...")

        try:
            # Check cache first
            dialogue_hash = self.get_dialogue_hash(step)
            cached_embeddings = self.get_cached_embeddings(dialogue_hash)
            
            if cached_embeddings is not None:
                print(f"[INFO] Using cached embeddings for step {step} (saved {time.time() - start_total:.2f}s)")
                return cached_embeddings
            
            # No cache hit, proceed with processing
            # Get the dialogue community report for this step
            dialogue_community_path = self.get_latest_dialogue_report(step)
            if not dialogue_community_path:
                print(f"[ERROR] No dialogue community report found for step {step}")
                return None
                
            # Load community data (big_communities is already cached in constructor)
            start_time = time.time()
            big_communities = self._big_communities_cache
            
            try:
                dialogue_communities = pd.read_parquet(dialogue_community_path)
                # Process only essential columns for dialogue communities
                essential_cols = ['community', 'title', 'summary', 'findings']
                cols_to_keep = [col for col in essential_cols if col in dialogue_communities.columns]
                dialogue_communities = dialogue_communities[cols_to_keep].copy()
            except Exception as e:
                print(f"Error loading dialogue communities: {e}")
                return None
                
            print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
                
            # Check if data was loaded successfully
            if big_communities.empty or dialogue_communities.empty:
                print("[WARNING] Failed to load community data properly")
                return None
            
            # Find similar communities
            start_time = time.time()
            similar_communities = self.find_similar_communities(
                big_communities, dialogue_communities, threshold=0.75, top_n=5)
            print(f"Community matching completed in {time.time() - start_time:.2f} seconds")
            
            # If no similar communities found, return None
            if not similar_communities:
                print("[WARNING] No similar communities found for this dialogue!")
                return None
            
            # Extract content from similar communities
            start_time = time.time()
            community_content = self.extract_community_content(big_communities, similar_communities)
            print(f"Content extraction completed in {time.time() - start_time:.2f} seconds")
            
            # Generate embeddings from community content
            start_time = time.time()
            community_embeddings = self.get_community_embeddings(community_content)
            print(f"Embedding generation completed in {time.time() - start_time:.2f} seconds")
            
            # Save to cache for future use
            self.save_embeddings_to_cache(dialogue_hash, community_embeddings)
            
            # Final stats
            print(f"[SUCCESS] Processing completed in {time.time() - start_total:.2f} seconds total")
            print(f"[DEBUG] Extracted Community Embeddings Shape: {community_embeddings.shape}")
            print(f"[DEBUG] Found {len(similar_communities)} similar communities")
            
            return community_embeddings
            
        except Exception as e:
            print(f"[ERROR] An error occurred in get_enhanced_rec_prompt: {e}")
            import traceback
            traceback.print_exc()
            return None

# Example usage
if __name__ == "__main__":
    enhancer = CommunityRecommenderPromptEnhancer()
    step = 5  # Example step number
    embeddings = enhancer.get_enhanced_rec_prompt(step)
    print(f"Got embeddings with shape: {embeddings.shape if embeddings is not None else 'None'}")