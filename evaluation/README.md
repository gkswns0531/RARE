# RARE Retrieval Evaluation System

## Overview

RARE provides a reproducible workflow for retrieval evaluation. The scripts in this directory load prebuilt datasets, run a configurable set of retrieval models, and compute group-based metrics.

## Key Features

- Batch execution for original queries
- Embedding cache reuse for corpus and query representations
- Group-level metrics covering coverage, perfect match, ranking quality, and reciprocal rank
- Optional comparison mode for previously generated result files
- Shell helper for sequential execution across multiple models

## RARE Evaluation Results

### Finance
| Model | Coverage@10 | Top@10 | NDCG@10 | MRR |
|-------|-------------|--------|---------|-----|
| BM25 | 59.97 | 36.28 | 48.57 | 36.17 |
| OpenAI-Large | 67.67 | 42.95 | 56.27 | 42.07 |
| BGE-M3 (0.56B) | 64.43 | 39.84 | 52.59 | 39.25 |
| Qwen3 (0.6B) | 67.62 | 43.80 | 54.85 | 40.39 |
| Qwen3 (4B) | 72.43 | 48.84 | 59.35 | 44.00 |
| Qwen3 (8B) | 72.92 | 47.44 | 60.35 | 44.78 |
| Gemma (0.3B) | 40.41 | 20.78 | 30.54 | 21.81 |
| Jina-v4 (3.75B) | 65.10 | 40.31 | 55.17 | 42.20 |
| E5-Large (0.56B) | 62.83 | 37.98 | 51.96 | 39.43 |
| E5-Mistral (7B) | 69.69 | 44.81 | 56.75 | 42.69 |

### Patent
| Model | Coverage@10 | Top@10 | NDCG@10 | MRR |
|-------|-------------|--------|---------|-----|
| BM25 | 76.35 | 55.62 | 63.15 | 48.78 |
| OpenAI-Large | 81.96 | 61.46 | 66.46 | 49.75 |
| BGE-M3 (0.56B) | 81.46 | 59.58 | 67.37 | 51.04 |
| Qwen3 (0.6B) | 81.49 | 60.21 | 68.03 | 51.45 |
| Qwen3 (4B) | 83.09 | 62.50 | 70.76 | 55.09 |
| Qwen3 (8B) | 84.05 | 63.12 | 71.38 | 54.59 |
| Gemma (0.3B) | 59.46 | 36.25 | 43.59 | 30.67 |
| Jina-v4 (3.75B) | 80.09 | 59.58 | 66.61 | 51.00 |
| E5-Large (0.56B) | 80.69 | 60.62 | 64.91 | 48.20 |
| E5-Mistral (7B) | 81.67 | 62.71 | 65.67 | 48.64 |

### Legal
| Model | Coverage@10 | Top@10 | NDCG@10 | MRR |
|-------|-------------|--------|---------|-----|
| BM25 | 49.08 | 27.77 | 40.25 | 29.93 |
| OpenAI-Large | 61.17 | 37.36 | 48.91 | 35.24 |
| BGE-M3 (0.56B) | 54.50 | 32.40 | 42.86 | 30.65 |
| Qwen3 (0.6B) | 60.29 | 35.37 | 46.64 | 33.08 |
| Qwen3 (4B) | 65.12 | 39.67 | 53.82 | 39.13 |
| Qwen3 (8B) | 67.16 | 41.49 | 57.08 | 42.51 |
| Gemma (0.3B) | 20.08 | 8.26 | 16.23 | 7.69 |
| Jina-v4 (3.75B) | 59.57 | 34.55 | 47.42 | 34.03 |
| E5-Large (0.56B) | 52.56 | 30.91 | 40.81 | 29.00 |
| E5-Mistral (7B) | 56.86 | 34.05 | 43.34 | 30.97 |

### Hotpot (Low Redundancy & Low Similarity)
| Model | Coverage@10 | Top@10 | NDCG@10 | MRR |
|-------|-------------|--------|---------|-----|
| BM25 | 64.58 | 44.19 | 61.80 | 48.63 |
| OpenAI-Large | 92.78 | 86.48 | 92.17 | 70.24 |
| BGE-M3 (0.56B) | 89.96 | 78.78 | 88.77 | 68.00 |
| Qwen3 (0.6B) | 93.97 | 89.18 | 92.93 | 70.49 |
| Qwen3 (4B) | 93.58 | 88.66 | 92.57 | 70.28 |
| Qwen3 (8B) | 93.58 | 88.66 | 92.57 | 70.28 |
| Gemma (0.3B) | 61.60 | 41.72 | 56.86 | 45.12 |
| Jina-v4 (3.75B) | 92.61 | 85.04 | 91.26 | 69.55 |
| E5-Large (0.56B) | 91.67 | 84.01 | 90.76 | 69.33 |
| E5-Mistral (7B) | 93.53 | 88.66 | 92.27 | 70.16 |

## Architecture

```
RARE/
├── dataset/                     # Shared evaluation datasets (JSON)
├── evaluation/
│   ├── run_evaluation.py        # Main evaluation script (original queries)
│   ├── models.py                # Core search functions (BM25, OpenAI, HuggingFace)
│   ├── metrics.py               # Group-based evaluation metrics
│   ├── run_all_models.sh        # Batch runner for multiple models/query types
│   ├── requirements.txt         # Evaluation-specific dependencies
│   ├── README.md                # Evaluation usage guide
│   ├── cache/                   # Default embedding cache (auto-created)
│   ├── finance_eval_dataset_cache/
│   ├── hotpot_eval_dataset_cache/
│   ├── legal_eval_dataset_cache/
│   ├── patent_eval_dataset_cache/
│   └── results/                 # Evaluation results written per run
└── ...
```

## Supported Models

### Traditional Models
- **BM25**: Statistical retrieval baseline

### OpenAI Models
- **OpenAI-Large**: `text-embedding-3-large` (3072-dim)

### HuggingFace Models
- **BGE-M3**: `BAAI/bge-m3` - Multilingual dense retrieval
- **BGE-Multivector**: `BAAI/bge-multivector` - FlagEmbedding multi-vector retriever with MaxSim scoring
- **Qwen3-0.6B**: `Qwen/Qwen3-Embedding-0.6B` - Compact model
- **Qwen3-4B**: `Qwen/Qwen3-Embedding-4B` - Large multilingual model  
- **Qwen3-8B**: `Qwen/Qwen3-Embedding-8B` - State-of-the-art model
- **Gemma-300M**: `google/embeddinggemma-300m` - Google's compact embedding model
- **Jina-v4**: `jinaai/jina-embeddings-v4` - Multimodal embedding model with retrieval task support
- **E5-Large**: `intfloat/multilingual-e5-large` - Multilingual E5 large model
- **E5-Mistral-7B**: `intfloat/e5-mistral-7b-instruct` - E5 model based on Mistral-7B with web search query support

## Evaluation Metrics

All metrics are **group-based**, focusing on information type completeness rather than individual chunk retrieval. Each group represents a distinct information type/category that must be retrieved for complete question answering.

### Core Metrics
- **coverage@k**: Average proportion of information types covered in top-k results (0.0-1.0)
- **perfect_match@k**: Binary metric - 1.0 if ALL information types are found in top-k, 0.0 otherwise  
- **ndcg@k**: NDCG computed at group level, rewarding early discovery of information types
- **mrr**: Mean Reciprocal Rank of first relevant chunk from any information type

### Metric Interpretation
- **k=3,5,10**: Different cutoff points for practical retrieval scenarios
- **Group Logic**: Each information type contributes at most once to the score
- **Complete vs Partial**: Perfect match requires 100% coverage, coverage allows partial success

**Example**: For gold groups `[(A,B), C, D]` representing 3 information types:
- Retrieved `[A, C, D]` → coverage@10=1.0, perfect_match@10=1.0 ✅
- Retrieved `[B, C, D]` → coverage@10=1.0, perfect_match@10=1.0 ✅  
- Retrieved `[A, B, C]` → coverage@10=0.67, perfect_match@10=0.0 ❌ (missing type D)


## Dataset Generation

Before running evaluations, you need to generate the evaluation dataset from RARE outputs.

### Prerequisites
Your RARE pipeline should have completed and generated final outputs in `outputs/` (default) or a directory passed via `--output-dir`.

### Generate Evaluation Dataset
```bash
# Generate from RARE final outputs
python evaluation/dataset_builder.py --input ../outputs/final_output/ --output ../dataset/finance_eval_dataset.json

# Or specify specific files
python evaluation/dataset_builder.py --files ../outputs/final_output/final_001.json ../outputs/final_output/final_004.json --output ../dataset/finance_eval_dataset.json

# Verify dataset statistics
python evaluation/dataset_builder.py --input ../outputs/final_output/ --stats-only
```


## Quick Start

### Basic Usage
```bash
# Single model evaluation (original queries)
python run_evaluation.py --models bm25

# Multiple models  
python run_evaluation.py --models bm25 openai_large

# All models with GPU acceleration
python run_evaluation.py --models all --device cuda

# Include BGE-Multivector with existing models
python run_evaluation.py --models bm25 bge_multivector qwen3_8b

### Notes on BGE-Multivector
- Uses FlagEmbedding APIs to produce multi-vector representations (`colbert_vecs`).
- MaxSim aggregation sums maximum token-to-token dot products per query token.
- Embedding caches are stored under `cache/flag_bge-multivector_*.pkl` for reuse across datasets.
```

### Custom Datasets
```bash
# Use different dataset file (auto-generates cache directory)
python run_evaluation.py --models bm25 --input-file ../dataset/finance_eval_dataset.json

# Specify custom cache directory
python run_evaluation.py --models bm25 --cache-dir cache/finance_eval_dataset

# Combine custom dataset with other query sets
python run_evaluation.py --models openai_large --input-file ../dataset/hotpot_eval_dataset.json
```

### API Usage
```python
from models import search, search_single

# Multiple queries (recommended)
results = search("openai_large", corpus, queries, batch_size=128)

# Single query
result = search_single("bge_m3", corpus, query, batch_size=128) 
```

## Configuration Options

### Command Line Arguments
```bash
--models MODELS [MODELS ...]
    Models to evaluate: bm25, openai_large, bge_m3, 
    qwen3_0.6b, qwen3_4b, qwen3_8b, gemma_embedding, jina_v4, e5_large, e5_mistral_7b, all

--input-file INPUT_FILE
    Custom dataset file path (default: ../dataset/finance_eval_dataset.json)

--cache-dir CACHE_DIR
    Custom cache directory (default: auto-generated from input filename)

--batch-size BATCH_SIZE
    Batch size for HuggingFace models (default: 16). OpenAI models use fixed batch size of 128.

--top-k TOP_K
    Number of top results to retrieve for each query (default: 10).

--device DEVICE
    Device for HuggingFace models: cpu, cuda (default: cpu)
```

### Environment Variables
```bash
# Required for OpenAI models
export OPENAI_API_KEY="your_openai_api_key"
```

## Performance Optimizations

### True Batch Processing
- **Decomposed Queries**: 2400x speedup - flattens all subqueries into single batch API call
  - Before: 800 samples × 3 subqueries = 2400 individual API calls
  - After: 1 batch API call for all 2400 subqueries
- **Fair Round-Robin Allocation**: K÷N allocation per subquery + round-robin for remaining slots
- **Smart Reconstruction**: Maintains per-sample grouping while maximizing batch efficiency

### Automatic Caching
- **Flexible Cache Management**: Auto-generated cache directories based on dataset filename
  - `eval_dataset.json` → `cache/`
  - `custom_dataset.json` → `custom_dataset_cache/`  
  - Any custom file → `{filename}_cache/`
- **Corpus embeddings**: Cached after first computation, reused across evaluations
- **Query embeddings**: Caching for original queries (dataset-specific)
- **Model instances**: HuggingFace models cached in memory to avoid reloading

### Batch Processing
- **OpenAI Models**: Fixed optimal batch size of 128 (memory-independent API calls)
- **HuggingFace Models**: GPU-optimized batch processing with configurable memory management (default: 16)
- **BM25**: Vectorized operations for instant bulk processing

### Memory Efficiency  
- **On-demand loading**: Models loaded only when needed
- **Efficient storage**: Compressed pickle format for embedding cache
- **Smart batching**: Automatic batch size adjustment based on model constraints

## Dataset Format

The evaluation system supports multiple dataset formats:

### Standard Dataset (`eval_dataset.json`)
```json
{
  "corpus": {
    "chunk_001": {
      "content": "Document content...",
      "metadata": {...}
    }
  },
  "queries": [
    {
      "question": "Query text?",
      "gold_chunk_ids": ["chunk_001", "chunk_002"],
      "gold_chunk_groups": [["chunk_001"], ["chunk_002"]]
    }
  ]
}
```

## API Reference

### Core Functions

#### `search(model_type, corpus, queries, top_k=10, batch_size=128)`
Main search interface for multiple queries.

**Parameters:**
- `model_type`: Model identifier (bm25, openai_large, etc.)
- `corpus`: Dictionary of chunk_id -> {content, metadata}  
- `queries`: List of query strings
- `top_k`: Number of results to return per query
- `batch_size`: Batch size for embedding models

**Returns:** `List[List[SearchResult]]`

#### `search_single(model_type, corpus, query, top_k=10, batch_size=128)`
Convenience function for single query.

**Returns:** `List[SearchResult]`

### Cache Management Functions

#### `set_cache_dir(cache_dir)`
Set global cache directory path.

**Parameters:**
- `cache_dir`: Path to cache directory

#### `get_cache_dir()`
Get current cache directory path.

**Returns:** `Path`

### SearchResult Class
```python
class SearchResult:
    chunk_id: str    # Document chunk identifier
    score: float     # Relevance score  
    rank: int        # Rank position (1-based)
```

## Requirements

```txt
See project-level requirements in ../requirements.txt
```

## Installation

```bash
# Clone repository and navigate to evaluation directory
cd RARE/evaluation/

# Install dependencies
pip install -r ../requirements.txt

# Set OpenAI API key (for OpenAI models)
export OPENAI_API_KEY="your_api_key_here"

# Run evaluation
python3 run_evaluation.py --models all
```

## Advanced Usage

### Custom Dataset Workflows
```bash
# Evaluate on custom dataset with auto-generated cache
python3 run_evaluation.py --models openai_large --input-file my_experiment.json

# Use specific cache directory for reproducibility
python3 run_evaluation.py --models bm25 --input-file test_data.json --cache-dir experiment_cache
```

## Technical Details

### Caching Strategy
- **Hash-based**: Corpus changes detected via MD5 hash of chunk IDs and count
- **Model-specific**: Separate cache files for each model type
- **Automatic cleanup**: Invalid caches detected and rebuilt automatically

### Batch Processing Logic
- **OpenAI**: Fixed batch size of 128 for optimal throughput (API-based, no memory constraints)
- **HuggingFace**: GPU memory-aware batching with automatic size adjustment
- **BM25**: Vectorized similarity computation across all queries simultaneously

### Error Handling
- **API failures**: Automatic retry with exponential backoff
- **Memory issues**: Graceful degradation to smaller batch sizes
- **Missing models**: Clear error messages with installation instructions

## Troubleshooting

### Common Issues

**OpenAI API Errors:**
```bash
# Check API key
echo $OPENAI_API_KEY

# Reduce batch size if hitting rate limits  
python run_evaluation.py --models openai_large --batch-size 32
```

**GPU Memory Issues:**
```bash
# Use CPU instead
python run_evaluation.py --models qwen3_8b --device cpu

# Or reduce batch size
python run_evaluation.py --models qwen3_8b --device cuda --batch-size 16
```

**Cache Issues:**
```bash
# Clear specific dataset cache
rm -rf custom_dataset_cache/
python run_evaluation.py --models bm25 --input-file custom_dataset.json

# Clear all caches
rm -rf cache/ *_cache/
python run_evaluation.py --models bm25  # Will rebuild cache

# Use custom cache directory
python run_evaluation.py --models bm25 --cache-dir fresh_cache
```

## Contributing

This evaluation system follows professional coding standards:
- Clean, intuitive API design
- Comprehensive error handling  
- Efficient caching and batch processing
- Clear documentation and examples

For questions or contributions, please refer to the main RARE documentation.