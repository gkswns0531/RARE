from enum import Enum
from typing import Optional


# =============================================================================
# Prompt Types for RARE
# =============================================================================
class PromptType(Enum):
    """Prompt types for RARE Framework"""
    
    # Atomic information extraction
    EXTRACT_ATOMIC_INFO = "extract_atomic_info"
    

    # Atomic information validation - Vanilla approach
    VERIFY_ATOMIC_INFO_VANILLA = "verify_atomic_info_vanilla"
    VERIFY_ATOMIC_INFO_VANILLA_FEWSHOT = "verify_atomic_info_vanilla_fewshot"
    
    # Atomic information validation - Combined approach  
    VERIFY_ATOMIC_INFO_VALIDITY = "verify_atomic_info_validity"
    
    # Atomic information validation - Separate approach
    VERIFY_ATOMIC_INFO_VALIDITY_SEPARATE = "verify_atomic_info_validity_separate"
    VERIFY_ATOMIC_INFO_COMPLETENESS_SEPARATE = "verify_atomic_info_completeness_separate"  
    VERIFY_ATOMIC_INFO_SPECIFICITY_SEPARATE = "verify_atomic_info_specificity_separate"
    VERIFY_ATOMIC_INFO_CLARITY_SEPARATE = "verify_atomic_info_clarity_separate"
    VERIFY_ATOMIC_INFO_QUESTIONABILITY_SEPARATE = "verify_atomic_info_questionability_separate"
    
    # Atomic information validation - Direct ranking output
    VERIFY_ATOMIC_INFO_RANKING = "verify_atomic_info_ranking"
    VERIFY_ATOMIC_INFO_RANKING_SEPARATE_VALIDITY = "verify_atomic_info_ranking_separate_validity"
    VERIFY_ATOMIC_INFO_RANKING_SEPARATE_COMPLETENESS = "verify_atomic_info_ranking_separate_completeness"
    VERIFY_ATOMIC_INFO_RANKING_SEPARATE_SPECIFICITY = "verify_atomic_info_ranking_separate_specificity"
    VERIFY_ATOMIC_INFO_RANKING_SEPARATE_CLARITY = "verify_atomic_info_ranking_separate_clarity"
    VERIFY_ATOMIC_INFO_RANKING_SEPARATE_QUESTIONABILITY = "verify_atomic_info_ranking_separate_questionability"
    
    # Multi-hop question generation (HotPot style)
    GENERATE_MULTIHOP_QUESTION = "generate_multihop_question"
    
    # LLM-based information selection and multi-hop question generation
    GENERATE_MULTIHOP_QUESTION_WITH_LLM_SELECTION = "generate_multihop_question_with_llm_selection"
    
    # Question paraphrasing for diversity
    PARAPHRASE_QUESTIONS = "paraphrase_questions"
    
    # Query decomposition
    DECOMPOSE_QUERY = "decompose_query"
    ESTIMATE_DECOMPOSITION_COUNT = "estimate_decomposition_count"
    
    # Multi-hop question separate validation (4 separate calls - answerability removed)
    VALIDATE_MULTIHOP_QUESTIONS_CONNECTIVITY_SEPARATE = "validate_multihop_questions_connectivity_separate"
    VALIDATE_MULTIHOP_QUESTIONS_FLUENCY_SEPARATE = "validate_multihop_questions_fluency_separate"
    VALIDATE_MULTIHOP_QUESTIONS_ESSENTIALITY_SEPARATE = "validate_multihop_questions_essentiality_separate"
    VALIDATE_MULTIHOP_QUESTIONS_VALIDITY_SEPARATE = "validate_multihop_questions_validity_separate"
    
    # Logical consistency pre-filtering for questions (4 separate calls)
    FILTER_CONTEXT_ASSUMPTION = "filter_context_assumption"
    FILTER_CIRCULAR_DEFINITION = "filter_circular_definition"
    FILTER_INFORMATION_COMPLETENESS = "filter_information_completeness"
    FILTER_QUESTION_AMBIGUITY = "filter_question_ambiguity"
    
    # Information completeness pre-filtering for atomic info (single call)  
    FILTER_ATOMIC_INFORMATION_COMPLETENESS = "filter_atomic_information_completeness"
    
    
    # Question answerability verification
    VERIFY_QUESTION_ANSWERABILITY = "verify_question_answerability"
    
    # Redundancy detection
    DETECT_SEMANTIC_REDUNDANCY = "detect_semantic_redundancy"
    
    # Legacy prompts (Step 4+ features - not used in Step 3.5 pipeline)
    GENERATE_GOLD_RANKING_LEGACY = "generate_gold_ranking_legacy"
    DETECT_SEMANTIC_REDUNDANCY_LEGACY = "detect_semantic_redundancy_legacy"
    GENERATE_QUESTION_FROM_ATOMIC_LEGACY = "generate_question_from_atomic_legacy"
    GENERATE_ANSWER_FROM_ATOMIC_LEGACY = "generate_answer_from_atomic_legacy"


# =============================================================================
# Model Pricing - GPT-4.1 & GPT-5 series (USD per 1K tokens)
# =============================================================================
MODEL_PRICING = {
    # GPT-4.1 series (fine-tuning capable)
    "gpt41": {"input": 0.003, "output": 0.012},
    "gpt41_mini": {"input": 0.0008, "output": 0.0032},
    "gpt41_nano": {"input": 0.0002, "output": 0.0008},
    
    # GPT-5 series (latest inference models)
    "gpt5": {"input": 0.00125, "output": 0.01},
    "gpt5_mini": {"input": 0.00025, "output": 0.002},
    "gpt5_nano": {"input": 0.00005, "output": 0.0004},
    
    # Compatibility
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    
    # Direct mapping
    "gpt-4.1": {"input": 0.003, "output": 0.012},
    "gpt-4.1-mini": {"input": 0.0008, "output": 0.0032}, 
    "gpt-4.1-nano": {"input": 0.0002, "output": 0.0008},
    "gpt-5": {"input": 0.00125, "output": 0.01},
    "gpt-5-mini": {"input": 0.00025, "output": 0.002},
    "gpt-5-nano": {"input": 0.00005, "output": 0.0004},
}

# =============================================================================
# Pipeline Limits
# =============================================================================
MAX_API_CALLS = 20000
MAX_COST_USD = 10.0
MAX_LOOP_COUNT = 5000

# =============================================================================
# Model Settings - Using cheapest model for testing
# =============================================================================
DEFAULT_MODEL = "gpt5_nano"  # GPT-5 nano - cheapest model
LANGUAGE = "English"

# =============================================================================
# Exchange Rate
# =============================================================================
EXCHANGE_RATE = 1300.0  # USD to KRW

# =============================================================================
# Redundancy Detection Settings
# =============================================================================
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 0

# =============================================================================
# Question Generation Settings
# =============================================================================
DEFAULT_TARGET_COUNT = 10
MIN_ATOMIC_INFO_LENGTH = 10
MAX_SIMILAR_ITEMS_CHECK = 20

# =============================================================================
# Embedding Model
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI embedding model

# =============================================================================
# Atomic Information Validation Settings
# =============================================================================
# Keywords to exclude from atomic information (hard filter)
EXCLUDE_KEYWORDS = [
    # English terms
    "document", "page", "title", "section", "chapter", "appendix",
    "figure", "table", "reference", "index", "header", "footer",
    
    # Korean terms  
    "문서", "페이지", "제목", "섹션", "장", "절", "항", "부록",
    "그림", "표", "참조", "색인", "머리글", "바닥글", "출처",
    
    # Positional terms
    "위치", "번호", "순서", "목차", "차례"
]

# Validation thresholds
VALIDITY_CONFIDENCE_THRESHOLD = -1.0  # Minimum confidence for validity (disabled)
BATCH_VALIDATION_SIZE = 10  # Number of items to validate in one batch

# Redundancy detection defaults
DEFAULT_REDUNDANCY_MAX_SIMILAR_ITEMS = 512
DEFAULT_REDUNDANCY_MAX_WORKERS = 1024
DEFAULT_REDUNDANCY_TOP_K_PER_CHUNK = 3

# Step 7 defaults
DEFAULT_STEP7_GENERATION_MODEL = "gpt5"
DEFAULT_STEP7_FILTER_MODEL = "gpt5_nano"
DEFAULT_STEP7_VALIDATION_MODEL = "gpt5_nano"
DEFAULT_STEP7_PARAPHRASE_MODEL = "gpt5_nano"
DEFAULT_STEP7_ANSWERABILITY_MODEL = "gpt5_nano"
DEFAULT_STEP7_NUM_INFORMATION = 2
DEFAULT_STEP7_NUM_QUESTIONS = 10
DEFAULT_STEP7_NUM_SAMPLES = 10
DEFAULT_STEP7_INPUT_POOL_SIZE = 100
DEFAULT_STEP7_MAX_WORKERS = 128


