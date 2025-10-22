# RARE — Redundancy-Aware RAG Evaluation Dataset Generator

RARE is a step-by-step framework to generate redundancy-aware RAG evaluation data from raw PDFs to final JSON outputs.

## Key Features

- Independent step execution (from PDF parsing to evaluation item generation)
- Intermediate artifacts saved as JSON/PKL for reuse
- End-to-end cost tracking with model-wise breakdown
- Bulk OpenAI Embeddings and threaded LLM calls
- Retry with exponential backoff for API robustness
- Progress bar for long-running steps
- Top-K per-document filtering to reduce cost
- Atomic information extraction, redundancy mapping, and question generation

## Pipeline Steps

1) parsing → PDF to text
2) chunking → token-aware chunking
3) atomic_info_extraction → atomic information extraction
4) atomic_info_selection → quality-based selection (separate scoring + RRF)
5) embedding_similarity → OpenAI embeddings (bulk)
6) redundancy_detection → semantic redundancy mapping
7) data_generation → question generation

## Quickstart

### 1. Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
# Optional: adjust defaults in rare_const.py (models, thresholds, pricing) before running
```

### 2. Full Pipeline
```bash
python run_complete_pipeline.py --input ANNUAL_REPORT_NVIDIA_2024.pdf --target-count 10
python run_complete_pipeline.py --input examples/ --target-count 10
```

### 3. Run Specific Steps
```bash
python run_complete_pipeline.py --input examples/ --steps parsing chunking
python run_complete_pipeline.py --steps atomic_info_extraction atomic_info_selection embedding_similarity redundancy_detection
python run_complete_pipeline.py --steps data_generation --target-count 5
python run_complete_pipeline.py --input document.pdf --target-count 10
python run_complete_pipeline.py --steps redundancy_detection --top-k-per-doc 1
```

## Repository Layout

```text
RARE/
├── README.md
├── requirements.txt
├── run_complete_pipeline.py         # Orchestrates full or partial RARE pipeline
├── rare_const.py                    # Global constants (models, thresholds, config)
├── rare_entities.py                 # Dataclass definitions for pipeline settings/results
├── rare_core/
│   ├── rare_orchestration_service.py   # Core orchestration logic across all steps
│   ├── rare_document_processor.py      # PDF parsing utilities
│   ├── rare_json_parser_service.py     # JSON cleaning/validation helpers
│   ├── rare_llm_client_service.py      # LLM client with cost tracking
│   ├── rare_search_client_service.py   # Embedding construction and similarity search
│   ├── rare_prompt_maker_service.py    # Prompt assembly helpers
│   ├── rare_prompts_service.py         # Prompt templates
│   ├── rare_ranking_utils.py           # Ranking / aggregation utilities
│   └── ...                             # Other support modules (cost tracker, savers)
├── rare_steps/
│   ├── run_step1_pdf_parsing.py
│   ├── run_step2_text_chunking.py
│   ├── run_step3_atomic_extraction.py
│   ├── run_step4_best_selection.py
│   ├── run_step5_embedding.py
│   ├── run_step6_redundancy.py
│   └── run_step7_multihop_questions.py
├── outputs/                     # Default directory for generated artifacts
├── outputs_backup/              # Cached backup outputs for each step
├── examples/                    # Sample PDF inputs
└── rare_output/                 # Legacy output directory (optional)
```

### Module Overview

- `run_complete_pipeline.py`: CLI entry point supporting full or selective step execution with argument parsing, logging setup, and result summarisation.
- `rare_const.py`: Centralised default thresholds, model choices, pricing info, and other constants referenced across services.
- `rare_entities.py`: Dataclasses encapsulating step settings, cost summaries, pipeline results, and evaluation item schemas.
- `rare_core/rare_orchestration_service.py`: Main orchestration engine coordinating each pipeline stage, managing intermediate artefacts, retries, and multi-step logic.
- `rare_core/rare_llm_client_service.py`: Wrapper around the OpenAI API offering retry logic, JSON validation, and detailed cost tracking for LLM calls.
- `rare_core/rare_search_client_service.py`: Handles embedding generation in bulk, tracks token usage and cost for embedding calls, and exposes similarity search utilities.
- `rare_core/rare_prompt_maker_service.py` / `rare_prompts_service.py`: Compose prompts used in each stage (extraction, filtering, validation, question generation).
- `rare_core/rare_ranking_utils.py`: Aggregates ranking outputs (e.g., Reciprocal Rank Fusion) and provides helper functions for multi-criteria sorting.
- `rare_steps/*.py`: Convenience scripts mirroring each pipeline stage for step-by-step execution, primarily calling the orchestrator with dedicated settings.

### 4. Programmatic Usage
```python
from rare_core.rare_orchestration_service import run_rare_pipeline

result = run_rare_pipeline(
    input_path="document.pdf",
    target_count=5,
    model="gpt5_nano",
    steps=["parsing", "chunking", "atomic_info", "selection", "embedding", "redundancy", "evaluation"],
    output_dir="results",
    max_workers=16,
)

result = run_rare_pipeline(
    steps=["redundancy"],
    output_dir="results",
    max_workers=16,
    top_k_per_document=1,
)

print(f"Total cost: ${result.cost_summary.total_cost_usd:.6f}")
print(f"Generated questions: {len(result.evaluation_items)}")
```

### 5. Outputs
```bash
# Default output directory: outputs/
python run_complete_pipeline.py --input document.pdf --target-count 5 --output-dir outputs
```

### 6. Examples and Step Scripts
See `examples/` for structure (add your own small public-domain PDF).

For step-by-step execution, see `rare_steps/`:
```bash
python rare_steps/run_step1_pdf_parsing.py
python rare_steps/run_step2_text_chunking.py
python rare_steps/run_step3_atomic_extraction.py
python rare_steps/run_step4_best_selection.py
python rare_steps/run_step5_embedding.py
python rare_steps/run_step6_redundancy.py
python rare_steps/run_step7_multihop_questions.py
```

## Performance

### Bulk processing
- Embeddings: OpenAI native bulk processing (up to ~1000 texts per request)
- LLM calls: multi-threaded via ThreadPoolExecutor

### Fallback and retry
- Automatic retries up to 3 attempts
- Exponential backoff: 0.5s → 1s → 2s
- Basic invalid-response detection

### Progress bar
- Per-step progress (e.g., 1/400)
- Live cost and throughput
- File name, page count, and chunk count

```bash
# Progress bar sample output
Extracting atomic info: 100%|████████| 50/50 [00:45<00:00, 1.12chunks/s, Cost=$0.1078, Processed=50/50]
Creating embeddings: 100%|██████████| 1/1 [00:01<00:00, 1.74batch/s, Texts=357/357, Batch_Size=357]
Detecting redundancies: 100%|████████| 44/44 [02:15<00:00, 0.32atomic_info/s, Cost=$0.0234, Unique=32, Redundant=12]
```

### Tuning tips
```bash
# Increase parallel workers (default: 16)
--max-workers 32

# Larger chunk size for higher throughput
--chunk-size 1024

# Save cost with Top-K per document
--top-k-per-doc 1
--top-k-per-doc 2
--top-k-per-doc 3

# Adjust embedding batch size (code-level)
batch_size=500  # SearchClient._get_bulk_embeddings()
```

### Large-document examples
```bash
# Example large document processing
python run_complete_pipeline.py --input large_doc.pdf

# Process embeddings in one API call when possible
python run_complete_pipeline.py --steps embedding

# Save ~90% cost with Top-1 per document
python run_complete_pipeline.py --steps redundancy --top-k-per-doc 1
```

## Result schema

### RedundancyPipelineResult
- `atomic_info_map`: atomic information per document
- `redundancy_mapping`: redundancy mapping result
- `evaluation_items`: generated evaluation items
- `cost_summary`: cost information
- `statistics`: pipeline statistics

### RareEvaluationItem
- `question`
- `target_answer`
- `atomic_info`
- `redundancy_level` (0 = unique, >0 = redundant)
- `chunk_id`
- `question_type`
- `document_source`
- `similar_atomic_info_ids`

## Configuration

### rare_const.py configuration:
- `DEFAULT_MODEL`: default LLM model
- `DEFAULT_SIMILARITY_THRESHOLD`: redundancy detection threshold
- `DEFAULT_TARGET_COUNT`: default number of questions to generate
- `EMBEDDING_MODEL`: embedding model

## Use cases

- Insurance documents: detect redundant policy content
- Legal documents: identify overlapping clauses
- Technical manuals: analyze repeated instructions
- Educational materials: discover duplicated concepts

## Data Sample
```
{
  "sample_id": "llm_sample_004",
  "question": "What was the total compensation in 2023 for NVIDIA Corporation’s president and chief executive officer?",
  "answer": "$21,356,924",
  "connectivity_score": 0.92,
  "fluency_score": 0.89,
  "essentiality_score": 0.9,
  "validity_score": 0.9,
  "rrf_score": 3.1666666666666665,
  "atomic_info_list": [
    {
      "atomic_id": "ANNUAL_REPORT_NVIDIA_2024_page085_chunk002_atomic_001",
      "content": "In 2023, Jen-Hsun Huang's total compensation was $21,356,924.",
      "chunk_id": "ANNUAL_REPORT_NVIDIA_2024_page085_chunk002"
    },
    {
      "atomic_id": "ANNUAL_REPORT_NVIDIA_2024_page184_chunk001_atomic_001",
      "content": "Jen-Hsun Huang is the President and Chief Executive Officer of NVIDIA Corporation.",
      "chunk_id": "ANNUAL_REPORT_NVIDIA_2024_page184_chunk001"
    }
  ],
  "gold_chunks": [
    [
      "ANNUAL_REPORT_NVIDIA_2024_page068_chunk001",
      "ANNUAL_REPORT_NVIDIA_2024_page085_chunk002",
      "ANNUAL_REPORT_NVIDIA_2024_page090_chunk001",
      "ANNUAL_REPORT_NVIDIA_2024_page111_chunk002",
      "ANNUAL_REPORT_NVIDIA_2024_page136_chunk001"
    ],
    [
      "ANNUAL_REPORT_NVIDIA_2024_page068_chunk001",
      "ANNUAL_REPORT_NVIDIA_2024_page102_chunk001",
      "ANNUAL_REPORT_NVIDIA_2024_page184_chunk001",
      "ANNUAL_REPORT_NVIDIA_2024_page185_chunk001",
      "ANNUAL_REPORT_NVIDIA_2024_page186_chunk001"
    ]
  ]
}
```
