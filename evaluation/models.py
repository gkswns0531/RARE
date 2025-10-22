"""RARE Retrieval Models."""

import os
import pickle
import hashlib
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path


class SearchResult:
    """Search result containing chunk ID, relevance score, and rank."""
    
    def __init__(self, chunk_id: str, score: float, rank: int):
        self.chunk_id = chunk_id
        self.score = score
        self.rank = rank


# Global model cache to avoid reloading
_model_cache = {}


def _flag_model_key(model_name: str) -> str:
    name = model_name.lower()
    if "bge" in name and "m3" in name:
        return "BAAI/bge-m3"
    if "bge" in name and "multivector" in name:
        return "BAAI/bge-m3"
    return model_name


# === CORE UTILITY FUNCTIONS ===

# Global cache directory setting
_cache_dir = None

def set_cache_dir(cache_dir: str):
    """Set cache directory path"""
    global _cache_dir
    _cache_dir = Path(cache_dir)

def get_cache_dir() -> Path:
    """Get current cache directory"""
    return _cache_dir or (Path(__file__).parent / "cache")

def get_cache_path(model_type: str, corpus_hash: str) -> Path:
    """Generate cache file path for model embeddings."""
    cache_dir = get_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{model_type}_{corpus_hash}.pkl"


def _get_flag_model(model_name: str):
    """Load FlagEmbedding model with caching."""
    resolved_name = _flag_model_key(model_name)
    if resolved_name not in _model_cache:
        from FlagEmbedding import FlagModel, BGEM3FlagModel
        use_fp16 = torch.cuda.is_available()
        if "bge" in resolved_name.lower():
            model = BGEM3FlagModel(resolved_name, use_fp16=use_fp16)
        else:
            model = FlagModel(resolved_name, use_fp16=use_fp16)
        _model_cache[resolved_name] = model
    return _model_cache[resolved_name]


def _extract_colbert_vectors(encode_output: Any) -> np.ndarray:
    """Extract colbert vectors from FlagEmbedding encode output."""
    if isinstance(encode_output, dict) and 'colbert_vecs' in encode_output:
        return np.asarray(encode_output['colbert_vecs'])
    if isinstance(encode_output, np.ndarray):
        return encode_output
    raise ValueError("FlagEmbedding output missing colbert_vecs")


def get_corpus_hash(corpus: Dict[str, Any]) -> str:
    """Generate hash for corpus to detect changes."""
    corpus_str = str(sorted(corpus.keys())) + str(len(corpus))
    return hashlib.md5(corpus_str.encode()).hexdigest()[:8]


def _hash_list_strings(values: List[str]) -> str:
    """Generate hash for list of strings."""
    joined = "||".join(values)
    return hashlib.md5(joined.encode()).hexdigest()[:12]


def save_cache(model_type: str, corpus_hash: str, data: Dict[str, Any]):
    """Save embeddings to cache."""
    cache_file = get_cache_path(model_type, corpus_hash)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


def load_cache(model_type: str, corpus_hash: str) -> Dict[str, Any]:
    """Load embeddings from cache if available."""
    cache_file = get_cache_path(model_type, corpus_hash)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    return None


def _save_flag_cache(model_type: str, corpus_hash: str, data: Dict[str, Any]):
    """Save FlagEmbedding vectors to cache."""
    cache_file = get_cache_path(model_type, corpus_hash)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


def _load_flag_cache(model_type: str, corpus_hash: str) -> Optional[Dict[str, Any]]:
    """Load FlagEmbedding vectors from cache if available."""
    cache_file = get_cache_path(model_type, corpus_hash)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def prepare_corpus(corpus: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Extract chunk IDs and texts from corpus."""
    chunk_ids = list(corpus.keys())
    texts = [corpus[cid]['content'] for cid in chunk_ids]
    return chunk_ids, texts


def create_search_results(chunk_ids: List[str], scores: np.ndarray, top_k: int) -> List[SearchResult]:
    """Create SearchResult objects from scores."""
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        results.append(SearchResult(
            chunk_id=chunk_ids[idx],
            score=float(scores[idx]),
            rank=rank + 1
        ))
    
    return results


def create_multi_search_results(chunk_ids: List[str], similarity_matrix: np.ndarray, top_k: int) -> List[List[SearchResult]]:
    """Create multiple SearchResult objects from similarity matrix."""
    all_results = []
    for i, scores in enumerate(similarity_matrix):
        results = create_search_results(chunk_ids, scores, top_k)
        all_results.append(results)
    
    return all_results


def _create_search_results_flag(chunk_ids: List[str], scores: np.ndarray, top_k: int) -> List[SearchResult]:
    """Create search results for flag multi-vector scores."""
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices):
        results.append(SearchResult(
            chunk_id=chunk_ids[idx],
            score=float(scores[idx]),
            rank=rank + 1
        ))
    return results


# === BM25 SEARCH ===

def search_bm25(corpus: Dict[str, Any], queries: List[str], top_k: int = 10) -> List[List[SearchResult]]:
    """Search using BM25 retrieval model."""
    from rank_bm25 import BM25Okapi
    
    chunk_ids, texts = prepare_corpus(corpus)
    tokenized = [text.split() for text in texts]
    
    bm25 = BM25Okapi(tokenized)
    
    all_results = []
    total_queries = len(queries)
    
    if total_queries > 100:
        print(f"Processing {total_queries} queries with BM25...")
        import sys
    
    for i, query in enumerate(queries):
        # Show progress for BM25 when processing many queries
        if total_queries > 100 and i % 50 == 0:
            percent = (i + 1) * 100 / total_queries
            filled_length = int(30 * (i + 1) // total_queries)
            bar = '█' * filled_length + '-' * (30 - filled_length)
            sys.stdout.write(f'\rBM25 processing |{bar}| {percent:.1f}% ({i + 1}/{total_queries})')
            sys.stdout.flush()
        
        scores = bm25.get_scores(query.split())
        results = create_search_results(chunk_ids, scores, top_k)
        all_results.append(results)
    
    if total_queries > 100:
        # Complete the progress bar
        sys.stdout.write(f'\rBM25 processing |{"█" * 30}| 100.0% ({total_queries}/{total_queries})')
        sys.stdout.flush()
        print()  # New line after progress bar
    
    return all_results


# === OPENAI SEARCH ===

def get_or_create_openai_embeddings(corpus: Dict[str, Any], model: str, batch_size: int = 128) -> tuple[List[str], np.ndarray]:
    """Get or create OpenAI embeddings with caching."""
    import openai
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    corpus_hash = get_corpus_hash(corpus)
    model_type = f"openai_{model.split('-')[-1]}"
    
    cache_data = load_cache(model_type, corpus_hash)
    
    if cache_data:
        return cache_data['chunk_ids'], cache_data['embeddings']
    
    # Create new embeddings
    chunk_ids, texts = prepare_corpus(corpus)
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_response = client.embeddings.create(input=batch_texts, model=model)
        batch_embeddings = [d.embedding for d in batch_response.data]
        all_embeddings.extend(batch_embeddings)
    
    corpus_embeddings = np.array(all_embeddings)
    
    save_cache(model_type, corpus_hash, {
        'chunk_ids': chunk_ids,
        'embeddings': corpus_embeddings
    })
    
    return chunk_ids, corpus_embeddings


def get_or_create_openai_query_embeddings(queries: List[str], model: str, batch_size: int = 128) -> np.ndarray:
    """Get or create OpenAI query embeddings with individual caching."""
    import hashlib
    import openai
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_type = f"openai_{model.split('-')[-1]}_queries"
    
    all_embeddings = []
    queries_to_compute = []
    cache_map = {}
    
    # Check cache for each individual query
    for i, query in enumerate(queries):
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:12]
        cache_file = get_cache_path(model_type, query_hash)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_embedding = pickle.load(f)
                all_embeddings.append(cached_embedding)
                cache_map[i] = len(all_embeddings) - 1
            except:
                # If cache is corrupted, add to compute list
                queries_to_compute.append((i, query))
        else:
            queries_to_compute.append((i, query))
    
    # Compute missing embeddings in batch
    if queries_to_compute:
        batch_queries = [q for _, q in queries_to_compute]
        query_batch_size = min(2048, batch_size * 4)
        
        if len(queries_to_compute) > 100:
            print(f"Computing embeddings for {len(queries_to_compute)} new queries...")
        
        computed_embeddings = []
        total_batches = (len(batch_queries) + query_batch_size - 1) // query_batch_size
        
        for batch_idx in range(0, len(batch_queries), query_batch_size):
            current_batch_num = batch_idx // query_batch_size + 1
            batch = batch_queries[batch_idx:batch_idx + query_batch_size]
            
            if len(queries_to_compute) > 100 and total_batches > 1:
                print(f"  Batch {current_batch_num}/{total_batches} ({len(batch)} queries)", end='\r')
            
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [d.embedding for d in response.data]
            computed_embeddings.extend(batch_embeddings)
        
        if len(queries_to_compute) > 100:
            print()  # New line after progress
        
        # Save individual caches and build result
        for j, (orig_idx, query) in enumerate(queries_to_compute):
            embedding = computed_embeddings[j]
            
            # Save to individual cache
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:12]
            cache_file = get_cache_path(model_type, query_hash)
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            
            cache_map[orig_idx] = len(all_embeddings)
            all_embeddings.append(embedding)
    
    # Reconstruct in original order
    final_embeddings = [None] * len(queries)
    for i in range(len(queries)):
        final_embeddings[i] = all_embeddings[cache_map[i]]
    
    return np.array(final_embeddings)


def search_openai(corpus: Dict[str, Any], queries: List[str], model: str = "text-embedding-3-small", top_k: int = 10, batch_size: int = 128) -> List[List[SearchResult]]:
    """Search using OpenAI embedding models with full caching."""
    
    # Get corpus embeddings (cached)
    chunk_ids, corpus_embeddings = get_or_create_openai_embeddings(corpus, model, batch_size)
    
    # Get query embeddings (cached)
    query_embeddings = get_or_create_openai_query_embeddings(queries, model, batch_size)
    
    # Compute cosine similarities (no API calls needed if both cached!)
    similarity_matrix = np.dot(query_embeddings, corpus_embeddings.T)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    query_norms = np.linalg.norm(query_embeddings, axis=1)
    similarity_matrix = similarity_matrix / (query_norms[:, np.newaxis] * corpus_norms[np.newaxis, :])
    
    return create_multi_search_results(chunk_ids, similarity_matrix, top_k)


# === HUGGINGFACE SEARCH ===

def get_or_create_huggingface_embeddings(corpus: Dict[str, Any], model_name: str, batch_size: int = 128) -> tuple[List[str], np.ndarray]:
    """Get or create HuggingFace embeddings with caching."""
    from sentence_transformers import SentenceTransformer
    
    corpus_hash = get_corpus_hash(corpus)
    model_type = model_name.split('/')[-1].lower()
    
    cache_data = load_cache(model_type, corpus_hash)
    
    if cache_data:
        return cache_data['chunk_ids'], cache_data['embeddings']
    
    # Create new embeddings
    if model_name not in _model_cache:
        # Memory-efficient loading for all models  
        if "embeddinggemma" in model_name.lower():
            model_kwargs = {
                "device_map": "auto", 
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True
            }
            model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
        elif "jina-embeddings-v4" in model_name.lower():
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs=model_kwargs)
        else:
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
        
        _model_cache[model_name] = model
    
    model = _model_cache[model_name]
    chunk_ids, texts = prepare_corpus(corpus)
    
    # Different models use different parameters  
    if "embeddinggemma" in model_name.lower():
        corpus_embeddings = model.encode_document(texts, batch_size=batch_size, show_progress_bar=False)
    elif "jina-embeddings-v4" in model_name.lower():
        corpus_embeddings = model.encode(texts, task="retrieval", prompt_name="passage", batch_size=batch_size, show_progress_bar=False)
    elif "e5-mistral-7b-instruct" in model_name.lower():
        corpus_embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    else:
        corpus_embeddings = model.encode(texts, prompt_name="document", batch_size=batch_size, show_progress_bar=False)
    
    save_cache(model_type, corpus_hash, {
        'chunk_ids': chunk_ids,
        'embeddings': corpus_embeddings
    })
    
    return chunk_ids, corpus_embeddings


def get_or_create_query_embeddings(queries: List[str], model_name: str, batch_size: int = 128) -> np.ndarray:
    """Get or create query embeddings with individual caching."""
    import hashlib
    from sentence_transformers import SentenceTransformer
    
    model_type = model_name.split('/')[-1].lower()
    
    all_embeddings = []
    queries_to_compute = []
    cache_map = {}
    
    # Check cache for each individual query
    for i, query in enumerate(queries):
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:12]
        cache_file = get_cache_path(f"{model_type}_queries", query_hash)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_embedding = pickle.load(f)
                all_embeddings.append(cached_embedding)
                cache_map[i] = len(all_embeddings) - 1
            except:
                queries_to_compute.append((i, query))
        else:
            queries_to_compute.append((i, query))
    
    # Compute missing embeddings in batch
    if queries_to_compute:
        if model_name not in _model_cache:
            # Memory-efficient loading for all models  
            if "embeddinggemma" in model_name.lower():
                model_kwargs = {
                    "device_map": "auto", 
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True
                }
                model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
            elif "jina-embeddings-v4" in model_name.lower():
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True
                }
                model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs=model_kwargs)
            else:
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True
                }
                model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
            
            _model_cache[model_name] = model
        
        model = _model_cache[model_name]
        batch_queries = [q for _, q in queries_to_compute]
        
        if len(queries_to_compute) > 100:
            print(f"Computing embeddings for {len(queries_to_compute)} new queries using {model_name.split('/')[-1]}...")
        
        # Different models use different parameters
        if "embeddinggemma" in model_name.lower():
            computed_embeddings = model.encode_query(batch_queries, batch_size=batch_size, show_progress_bar=False)
        elif "jina-embeddings-v4" in model_name.lower():
            computed_embeddings = model.encode(batch_queries, task="retrieval", prompt_name="query", batch_size=batch_size, show_progress_bar=False)
        elif "e5-mistral-7b-instruct" in model_name.lower():
            computed_embeddings = model.encode(batch_queries, prompt_name="web_search_query", batch_size=batch_size, show_progress_bar=False)
        else:
            computed_embeddings = model.encode(batch_queries, prompt_name="query", batch_size=batch_size, show_progress_bar=False)
        
        # Save individual caches and build result
        for j, (orig_idx, query) in enumerate(queries_to_compute):
            embedding = computed_embeddings[j]
            
            # Save to individual cache
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:12]
            cache_file = get_cache_path(f"{model_type}_queries", query_hash)
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            
            cache_map[orig_idx] = len(all_embeddings)
            all_embeddings.append(embedding)
    
    # Reconstruct in original order
    final_embeddings = [None] * len(queries)
    for i in range(len(queries)):
        final_embeddings[i] = all_embeddings[cache_map[i]]
    
    return np.array(final_embeddings)


def search_huggingface(corpus: Dict[str, Any], queries: List[str], model_name: str, top_k: int = 10, batch_size: int = 128) -> List[List[SearchResult]]:
    """Search using HuggingFace models with full caching."""
    
    # Get corpus embeddings (cached)
    chunk_ids, corpus_embeddings = get_or_create_huggingface_embeddings(corpus, model_name, batch_size)
    
    # Get query embeddings (cached)
    query_embeddings = get_or_create_query_embeddings(queries, model_name, batch_size)
    
    # Compute cosine similarities (no model loading needed if both cached!)
    from sentence_transformers.util import cos_sim
    similarity_matrix = cos_sim(query_embeddings, corpus_embeddings).cpu().numpy()
    
    return create_multi_search_results(chunk_ids, similarity_matrix, top_k)


def _get_or_create_flag_embeddings(corpus: Dict[str, Any], model_name: str, batch_size: int = 32) -> tuple[List[str], List[np.ndarray]]:
    """Get or create FlagEmbedding multi-vector representations."""
    corpus_hash = get_corpus_hash(corpus)
    model_key = f"flag_{model_name.split('/')[-1].lower()}"

    cache_data = _load_flag_cache(model_key, corpus_hash)
    if cache_data:
        return cache_data['chunk_ids'], cache_data['embeddings']

    model = _get_flag_model(model_name)
    chunk_ids, texts = prepare_corpus(corpus)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        outputs = model.encode_corpus(batch_texts, return_colbert_vecs=True)
        if isinstance(outputs, dict) and 'colbert_vecs' in outputs:
            for vectors in outputs['colbert_vecs']:
                embeddings.append(np.asarray(vectors))
        else:
            for output in outputs:
                vectors = _extract_colbert_vectors(output)
                embeddings.append(vectors)

    _save_flag_cache(model_key, corpus_hash, {
        'chunk_ids': chunk_ids,
        'embeddings': embeddings
    })

    return chunk_ids, embeddings


def _get_or_create_flag_query_embeddings(queries: List[str], model_name: str, batch_size: int = 32) -> List[np.ndarray]:
    """Get or create FlagEmbedding query vectors."""
    model_key = f"flag_{model_name.split('/')[-1].lower()}"
    all_embeddings = []
    queries_to_compute = []
    cache_map = {}

    for idx, query in enumerate(queries):
        query_hash = _hash_list_strings([query])
        cache_file = get_cache_path(f"{model_key}_queries", query_hash)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                all_embeddings.append(cached)
                cache_map[idx] = len(all_embeddings) - 1
            except Exception:
                queries_to_compute.append((idx, query))
        else:
            queries_to_compute.append((idx, query))

    if queries_to_compute:
        model = _get_flag_model(model_name)
        batch_queries = [q for _, q in queries_to_compute]

        for batch_start in range(0, len(batch_queries), batch_size):
            sub_queries = batch_queries[batch_start:batch_start + batch_size]
            outputs = model.encode_queries(sub_queries, return_colbert_vecs=True)
            if isinstance(outputs, dict) and 'colbert_vecs' in outputs:
                vec_list = [np.asarray(v) for v in outputs['colbert_vecs']]
            else:
                vec_list = [_extract_colbert_vectors(o) for o in outputs]

            for offset, vectors in enumerate(vec_list):
                query_idx, query_text = queries_to_compute[batch_start + offset]
                query_hash = _hash_list_strings([query_text])
                cache_file = get_cache_path(f"{model_key}_queries", query_hash)
                with open(cache_file, 'wb') as f:
                    pickle.dump(vectors, f)
                cache_map[query_idx] = len(all_embeddings)
                all_embeddings.append(vectors)

    final_embeddings = [None] * len(queries)
    for idx in range(len(queries)):
        final_embeddings[idx] = all_embeddings[cache_map[idx]]

    return final_embeddings


def _maxsim_similarity(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Compute MaxSim similarity between query and document vectors."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    scores = np.dot(query_vecs, doc_vecs.T)
    max_scores = scores.max(axis=1)
    return float(max_scores.sum())


def _compute_flag_similarity_matrix(query_embeddings: List[np.ndarray], doc_embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute similarity matrix for FlagEmbedding multi-vector representations."""
    num_queries = len(query_embeddings)
    num_docs = len(doc_embeddings)
    similarity_matrix = np.zeros((num_queries, num_docs), dtype=np.float32)

    for qi in range(num_queries):
        q_vecs = query_embeddings[qi]
        for di in range(num_docs):
            d_vecs = doc_embeddings[di]
            similarity_matrix[qi, di] = _maxsim_similarity(q_vecs, d_vecs)

    return similarity_matrix


def search_flag_multivector(corpus: Dict[str, Any], queries: List[str], model_name: str, top_k: int = 10, batch_size: int = 32) -> List[List[SearchResult]]:
    """Search using FlagEmbedding multi-vector models."""
    chunk_ids, corpus_embeddings = _get_or_create_flag_embeddings(corpus, model_name, batch_size)
    query_embeddings = _get_or_create_flag_query_embeddings(queries, model_name, batch_size)

    similarity_matrix = _compute_flag_similarity_matrix(query_embeddings, corpus_embeddings)

    results = []
    for scores in similarity_matrix:
        results.append(_create_search_results_flag(chunk_ids, scores, top_k))

    return results


# === UNIFIED INTERFACE ===

def search(model_type: str, corpus: Dict[str, Any], queries: List[str], top_k: int = 10, batch_size: int = 128) -> List[List[SearchResult]]:
    """Unified search interface for all model types."""
    
    if model_type == "bm25":
        return search_bm25(corpus, queries, top_k)
    
    elif model_type == "openai_large":
        return search_openai(corpus, queries, "text-embedding-3-large", top_k, 128)
    
    elif model_type == "bge_m3":
        return search_huggingface(corpus, queries, "BAAI/bge-m3", top_k, batch_size)
    
    elif model_type == "qwen3_0.6b":
        return search_huggingface(corpus, queries, "Qwen/Qwen3-Embedding-0.6B", top_k, batch_size)
    
    elif model_type == "qwen3_4b":
        return search_huggingface(corpus, queries, "Qwen/Qwen3-Embedding-4B", top_k, batch_size)
    
    elif model_type == "qwen3_8b":
        return search_huggingface(corpus, queries, "Qwen/Qwen3-Embedding-8B", top_k, batch_size)
    
    elif model_type == "gemma_embedding":
        return search_huggingface(corpus, queries, "google/embeddinggemma-300m", top_k, batch_size)
    
    elif model_type == "jina_v4":
        return search_huggingface(corpus, queries, "jinaai/jina-embeddings-v4", top_k, batch_size)
    
    elif model_type == "e5_large":
        return search_huggingface(corpus, queries, "intfloat/multilingual-e5-large", top_k, batch_size)
    
    elif model_type == "e5_mistral_7b":
        return search_huggingface(corpus, queries, "intfloat/e5-mistral-7b-instruct", top_k, batch_size)

    elif model_type == "bge_multivector":
        return search_flag_multivector(corpus, queries, "BAAI/bge-multivector", top_k, batch_size)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def search_single(model_type: str, corpus: Dict[str, Any], query: str, top_k: int = 10, batch_size: int = 128) -> List[SearchResult]:
    """Single query search (wrapper around multi-query search)."""
    batch_results = search(model_type, corpus, [query], top_k, batch_size)
    return batch_results[0]


# === DECOMPOSED QUERY SEARCH ===

def get_decomposed_cache_path(model_type: str, corpus_hash: str, query_type: str) -> Path:
    """Generate cache file path for decomposed query embeddings."""
    cache_dir = get_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{model_type}_queries_{query_type}_{corpus_hash}.pkl"


def get_query_hash(queries: List[str]) -> str:
    """Generate hash for query list."""
    queries_str = "||".join(sorted(queries))
    return hashlib.md5(queries_str.encode()).hexdigest()[:8]


def save_query_cache(model_type: str, corpus_hash: str, query_type: str, data: Dict[str, Any]):
    """Save query embeddings to cache."""
    cache_file = get_decomposed_cache_path(model_type, corpus_hash, query_type)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


def load_query_cache(model_type: str, corpus_hash: str, query_type: str) -> Dict[str, Any]:
    """Load query embeddings from cache if available."""
    cache_file = get_decomposed_cache_path(model_type, corpus_hash, query_type)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    return None




def get_simple_name(model_type: str) -> str:
    """Get clean display name for model with parameter size."""
    names = {
        "bm25": "BM25",
        "openai_large": "OpenAI-Large", 
        "bge_m3": "BGE-M3 (0.56B)",
        "qwen3_0.6b": "Qwen3 (0.6B)",
        "qwen3_4b": "Qwen3 (4B)",
        "qwen3_8b": "Qwen3 (8B)",
        "gemma_embedding": "Gemma (0.3B)",
        "jina_v4": "Jina-v4 (3.75B)",
        "e5_large": "E5-Large (0.56B)",
        "e5_mistral_7b": "E5-Mistral (7B)",
        "bge_multivector": "BGE-Multivector (0.56B)"
    }
    return names.get(model_type, model_type)