#!/bin/bash

# RARE Sequential Model Evaluation Script - Original Queries Only
# Runs all models with standard queries (decomposed modes temporarily disabled)

echo "RARE Original Query Evaluation"
echo "==============================================="

# Configuration
BATCH_SIZE=16
RESULTS_DIR="results"
INPUT_FILE="../dataset/finance_eval_dataset.json"
QUERY_TYPES=("original")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# List of models to run individually (HF models that use GPU memory)
HF_MODELS=(
    "gemma_embedding"
    "e5_large"
    "e5_mistral_7b"
    "bge_m3"
    "jina_v4" 
    "qwen3_0.6b"
    "qwen3_4b"
    "qwen3_8b"
)

# Non-GPU intensive models (can run together)
LIGHT_MODELS="bm25 openai_large"

echo "Plan:"
echo "  - Dataset: $INPUT_FILE"
echo "  - Query types: ${QUERY_TYPES[*]}"
echo "  - Light models: $LIGHT_MODELS"
echo "  - HF models: ${HF_MODELS[*]}"
echo "  - Batch size: $BATCH_SIZE"
echo ""

# Function to run evaluation with retry (efficient: all query types at once)
run_evaluation_efficient() {
    local model=$1
    local attempt=1
    local max_attempts=2
    
    while [ $attempt -le $max_attempts ]; do
        echo "[$attempt/$max_attempts] Running: $model (original)"
        
        if python3 run_evaluation.py --models $model --batch-size $BATCH_SIZE --input-file $INPUT_FILE --query-type original; then
            echo "Success: $model (original)"
            return 0
        else
            echo "Failed: $model (original) (attempt $attempt)"
            if [ $attempt -lt $max_attempts ]; then
                echo "Waiting 10 seconds before retry..."
                sleep 10
            fi
        fi
        
        ((attempt++))
    done
    
    echo "Final failure: $model (original)"
    return 1
}

# Clear any existing GPU memory
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 5

# EFFICIENT MODE: Process all query types per model in single load
total_models=$(( ${#HF_MODELS[@]} + 1 ))
current_model=0

# Step 1: Run light models (all query types at once)
current_model=$((current_model + 1))
echo ""
echo "===== LIGHT MODELS (BM25 + OpenAI-Large) ====="
echo "=============================================="
echo "[$current_model/$total_models] Processing all ${#QUERY_TYPES[@]} query types for light models..."

if run_evaluation_efficient "$LIGHT_MODELS"; then
    echo "Light models completed (all query types)"
    
    # Show method comparison for each light model
    echo "Generating method comparison tables for light models..."
    python3 -c 'import sys; sys.path.append("."); from run_evaluation import print_method_comparison_tables; from pathlib import Path; input_file = "'"$INPUT_FILE"'"; dataset_name = Path(input_file).stem; light_models = ["bm25", "openai_large"]; [print_method_comparison_tables([model], dataset_name, 10) for model in light_models]'
else
    echo "Light models failed"
fi

# Step 2: Run each HF model (all query types at once per model)
for model in "${HF_MODELS[@]}"; do
    current_model=$((current_model + 1))
    
    echo ""
    echo "===== MODEL: $model ====="
    echo "============================="
    echo "[$current_model/$total_models] Processing all ${#QUERY_TYPES[@]} query types for $model..."
    
    # Clear GPU memory before loading new model
    echo "Clearing GPU memory for new model..."
    python3 -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')" 2>/dev/null || true
    sleep 5
    
    # Process ALL query types for this model in single run (EFFICIENT!)
    if run_evaluation_efficient "$model"; then
        echo "Completed: $model (all query types)"
        
        # Show method comparison for this specific model
        echo "Generating method comparison table for $model..."
        python3 -c 'import sys; sys.path.append("."); from run_evaluation import print_method_comparison_tables; from pathlib import Path; input_file = "'"$INPUT_FILE"'"; dataset_name = Path(input_file).stem; print_method_comparison_tables(["'"$model"'"], dataset_name, 10)'
    else
        echo "Failed: $model (all query types)"
    fi
    
    echo "Waiting for model cleanup..."
    sleep 5
done

echo ""
echo "COMPREHENSIVE EVALUATION COMPLETED!"
echo "======================================"
echo "Results summary:"
echo "  - Dataset: $INPUT_FILE"
echo "  - Query types evaluated: ${QUERY_TYPES[*]}"
echo "  - Models evaluated: $total_models"  
echo "  - Results directory: $RESULTS_DIR/"
echo "  - Evaluation approach: ULTRA-EFFICIENT (single load per model)"
echo ""
echo "Efficiency Gains:"
echo "  - Old approach: 36 separate runs (9 models × 4 query types)"
echo "  - New approach: 9 runs only (1 per model, all query types together)"
echo "  - Speed improvement: ~4x faster model loading!"
echo ""
echo "Generated result files:"
echo "  - evaluation_results_*.json"
echo ""
echo "Models evaluated:"
echo "  - Light models: $LIGHT_MODELS"
echo "  - HF models: ${HF_MODELS[*]}"
echo ""
echo "Performance Analysis Available:"
echo "  - Original query comparison across models"
echo "  - Model comparison across all query types"  
echo "  - Hop-based analysis (1-hop, 2-hop, 3-hop, 4-hop)"
echo ""
echo "Next Steps:"
echo "  - Check individual JSON files in $RESULTS_DIR/"
echo "  - Review Coverage@10, Top@10, NDCG@10, MRR across models"
echo "  - Review per-model metrics for original queries"
echo ""
echo "Ultra-efficient evaluation completed successfully!"
echo ""
echo ""
echo "=============================================================================="
echo "FINAL COMPREHENSIVE COMPARISON - ALL MODELS"
echo "=============================================================================="
echo ""
python3 -c 'import sys; sys.path.append("."); from run_evaluation import print_method_comparison_tables; from pathlib import Path; input_file = "'"$INPUT_FILE"'"; dataset_name = Path(input_file).stem; light_models = ["bm25", "openai_large"]; hf_models = "'"${HF_MODELS[*]}"'".split(); all_models = light_models + hf_models; print_method_comparison_tables(all_models, dataset_name, 10)'
