from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rare_const import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    DEFAULT_REDUNDANCY_MAX_SIMILAR_ITEMS,
    DEFAULT_REDUNDANCY_MAX_WORKERS,
    DEFAULT_REDUNDANCY_TOP_K_PER_CHUNK,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_STEP7_ANSWERABILITY_MODEL,
    DEFAULT_STEP7_FILTER_MODEL,
    DEFAULT_STEP7_GENERATION_MODEL,
    DEFAULT_STEP7_INPUT_POOL_SIZE,
    DEFAULT_STEP7_MAX_WORKERS,
    DEFAULT_STEP7_NUM_INFORMATION,
    DEFAULT_STEP7_NUM_QUESTIONS,
    DEFAULT_STEP7_NUM_SAMPLES,
    DEFAULT_STEP7_PARAPHRASE_MODEL,
    DEFAULT_STEP7_VALIDATION_MODEL,
    DEFAULT_TARGET_COUNT,
    LANGUAGE,
)


@dataclass
class AtomicInfo:
    """Atomic information unit with content and validation scores"""
    content: str
    chunk_id: str = ""
    atomic_info_id: str = ""
    
    # Validation scores (added for quality assessment)
    validity_score: float = 0.0        # 0.0-1.0: Business value and practical utility
    completeness_score: float = 0.0    # 0.0-1.0: Self-containment and context independence  
    specificity_score: float = 0.0     # 0.0-1.0: Concrete details and actionable precision
    clarity_score: float = 0.0         # 0.0-1.0: Unambiguous meaning and interpretation
    questionability_score: float = 0.0 # 0.0-1.0: Natural question generation potential
    overall_confidence: float = 0.0    # 0.0-1.0: Calculated average of all validation scores
    overall_quality_score: float = 0.0 # 0.0-1.0: Vanilla ablation study single quality score
    validation_reasoning: str = ""     # Detailed reasoning from LLM validation
    


@dataclass
class RedundancyMapping:
    """Redundancy mapping result for atomic information"""
    atomic_info_id: str
    content: str
    chunk_id: str
    redundant_items: List[str] = field(default_factory=list)  # List of redundant atomic_info_ids, or ["unique"] if unique
    similarity_scores: Dict[str, float] = field(default_factory=dict)  # atomic_info_id -> similarity_score


@dataclass
class RareDataSample:
    """Single data sample with question-answer pair and atomic info context"""
    question: str
    answer: str
    atomic_info: str  # The atomic information that led to this QA pair
    atomic_info_id: str
    redundancy_level: int  # 0 for unique, >0 for redundant items
    chunk_id: str


@dataclass
class RareEvaluationItem:
    """Evaluation item for RARE redundancy-aware benchmark"""
    question: str
    target_answer: str
    atomic_info: str
    atomic_info_id: str
    redundancy_level: int
    chunk_id: str
    question_type: str = "redundancy_aware"
    document_source: str = ""
    similar_atomic_info_ids: List[str] = field(default_factory=list)


@dataclass
class CostSummary:
    """Cost tracking for LLM API calls"""
    total_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_cost_krw: float = 0.0
    model_types: Dict[str, float] = field(default_factory=dict)


@dataclass
class RedundancyPipelineResult:
    """Complete result of redundancy detection pipeline"""
    atomic_info_map: Dict[str, List[AtomicInfo]]
    redundancy_mapping: Dict[str, RedundancyMapping]
    evaluation_items: List[RareEvaluationItem]
    cost_summary: CostSummary
    statistics: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Step1Settings:
    input_dir: str = "examples"
    output_dir: str = "outputs/step1"
    language: str = LANGUAGE


@dataclass(frozen=True)
class Step2Settings:
    input_file: str = "outputs/step1/parsed_documents.json"
    output_dir: str = "outputs/step2"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP


@dataclass(frozen=True)
class Step3Settings:
    input_file: str = "outputs/step2/text_chunks.json"
    output_dir: str = "outputs/step3"
    model: str = DEFAULT_MODEL
    workers: int = 1024
    language: str = LANGUAGE
    limit: Optional[int] = None


@dataclass(frozen=True)
class Step4Settings:
    input_file: str = "outputs/step3/atomic_info.json"
    output_dir: str = "outputs/step4"
    model: str = DEFAULT_MODEL
    workers: int = 1024
    language: str = LANGUAGE
    limit: Optional[int] = None
    logical_filtering_model: str = DEFAULT_MODEL
    skip_logical_filtering: bool = False


@dataclass(frozen=True)
class Step5Settings:
    input_file: str = "outputs/step4/atomic_info_filtered.json"
    output_dir: str = "outputs/step5"
    batch_size: int = 2048
    limit: Optional[int] = None
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    auto_threshold_batch_size: int = 1024
    embedding_only: bool = False
    similarity_only: bool = False
    similarity_batch_size: int = 32


@dataclass(frozen=True)
class Step6Settings:
    embeddings_file: str = "outputs/step5/step4_embeddings.pkl"
    output_dir: str = "outputs/step6"
    model: str = DEFAULT_MODEL
    language: str = LANGUAGE
    max_workers: int = DEFAULT_REDUNDANCY_MAX_WORKERS
    max_similar_items: int = DEFAULT_REDUNDANCY_MAX_SIMILAR_ITEMS
    top_k_per_chunk: int = DEFAULT_REDUNDANCY_TOP_K_PER_CHUNK
    max_chunks: Optional[int] = None


@dataclass(frozen=True)
class Step7Settings:
    input_file: str = "outputs/step6/redundancy_mapping.json"
    chunk_file: str = "outputs/step2/text_chunks.json"
    output_dir: str = "outputs/step7"
    generation_model: str = DEFAULT_STEP7_GENERATION_MODEL
    filter_model: str = DEFAULT_STEP7_FILTER_MODEL
    validation_model: str = DEFAULT_STEP7_VALIDATION_MODEL
    answerability_model: str = DEFAULT_STEP7_ANSWERABILITY_MODEL
    language: str = LANGUAGE
    num_information: int = DEFAULT_STEP7_NUM_INFORMATION
    num_questions: int = DEFAULT_STEP7_NUM_QUESTIONS
    num_samples: int = DEFAULT_STEP7_NUM_SAMPLES
    input_pool_size: int = DEFAULT_STEP7_INPUT_POOL_SIZE
    output_questions: int = DEFAULT_STEP7_NUM_QUESTIONS
    max_workers: int = DEFAULT_STEP7_MAX_WORKERS
    legacy_mode: bool = False


STEP1_SETTINGS = Step1Settings()
STEP2_SETTINGS = Step2Settings()
STEP3_SETTINGS = Step3Settings()
STEP4_SETTINGS = Step4Settings()
STEP5_SETTINGS = Step5Settings()
STEP6_SETTINGS = Step6Settings()
STEP7_SETTINGS = Step7Settings()
