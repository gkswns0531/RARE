from typing import Any, List, Dict
import openai
import numpy as np
from tqdm import tqdm
from rare_const import EMBEDDING_MODEL
import torch


class SearchClient:
    """OpenAI Embeddings based vector search client for RARE"""
    
    def __init__(self, documents: List[dict] = None, api_key: str = None, batch_size: int = 1024):
        """
        Initialize search client with OpenAI embeddings
        documents: List of documents with content and metadata
        batch_size: Batch size for embedding generation
        """
        self.documents = documents or []
        self.client = openai.OpenAI(api_key=api_key)
        self.doc_embeddings = None
        self.batch_size = batch_size
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.embedding_prices = {
            "text-embedding-3-small": 0.02 / 1_000_000,  # $0.02 per 1M tokens
            "text-embedding-3-large": 0.13 / 1_000_000   # $0.13 per 1M tokens
        }
        
        if self.documents:
            self._build_embeddings()
    
    def set_documents(self, documents: List[dict]):
        """Set documents and rebuild embeddings"""
        self.documents = documents
        self._build_embeddings()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for single text (for query)"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            
            # Track usage and cost
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            price_per_token = self.embedding_prices.get(EMBEDDING_MODEL, 0.0)
            cost = tokens_used * price_per_token
            self.total_cost_usd += cost
            
            return np.array(response.data[0].embedding)
        except Exception as e:
            raise Exception(f"Failed to get embedding: {str(e)}")
    
    def _get_bulk_embeddings(self, texts: List[str], batch_size: int = 1000) -> List[np.ndarray]:
        """Get OpenAI embeddings for multiple texts in bulk with progress bar"""
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches to avoid API limits with progress bar
        with tqdm(total=total_batches, desc="Creating embeddings", unit="batch") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    response = self.client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch_texts
                    )
                    
                    # Track usage and cost
                    tokens_used = response.usage.total_tokens
                    self.total_tokens_used += tokens_used
                    
                    price_per_token = self.embedding_prices.get(EMBEDDING_MODEL, 0.0)
                    cost = tokens_used * price_per_token
                    self.total_cost_usd += cost
                    
                    batch_embeddings = [np.array(item.embedding) for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    pbar.set_postfix({
                        'Texts': f"{len(all_embeddings)}/{len(texts)}",
                        'Batch_Size': len(batch_texts),
                        'Cost': f"${self.total_cost_usd:.4f}"
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    raise Exception(f"Failed to get bulk embeddings for batch {i//batch_size + 1}: {str(e)}")
        
        return all_embeddings
    
    def _build_embeddings(self):
        """Build embeddings for all documents using bulk processing"""
        if not self.documents:
            return
        
        print(f"Building embeddings for {len(self.documents)} documents (bulk processing)...")
        
        # Extract texts from documents  
        texts = [doc.get("content", "") for doc in self.documents]
        
        # Generate embeddings in bulk
        embeddings = self._get_bulk_embeddings(texts, self.batch_size)
        
        self.doc_embeddings = np.array(embeddings)
        print(f"Completed bulk embedding generation: {self.doc_embeddings.shape}")
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        if not self.documents or self.doc_embeddings is None:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate cosine similarities
        if torch.cuda.is_available():
            doc_tensor = torch.from_numpy(self.doc_embeddings).cuda()
            query_tensor = torch.from_numpy(query_embedding).cuda()
            similarities = torch.matmul(doc_tensor, query_tensor).cpu().numpy()
        else:
            similarities = np.dot(self.doc_embeddings, query_embedding)
        
        # Sort by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Build results
        results = []
        for idx in sorted_indices[:top_k]:
            similarity_score = similarities[idx]
            if similarity_score > 0.0:  # Only return positive similarities
                doc = self.documents[idx].copy()
                doc["score"] = float(similarity_score)
                results.append(doc)
        
        return results
    
    def get_total_cost_usd(self) -> float:
        """Get total cost of embedding operations in USD"""
        return self.total_cost_usd
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": self.total_cost_usd,
            "model": EMBEDDING_MODEL,
            "price_per_1m_tokens": self.embedding_prices.get(EMBEDDING_MODEL, 0.0) * 1_000_000
        }
