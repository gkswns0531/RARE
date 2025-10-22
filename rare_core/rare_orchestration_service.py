import logging
import random
import json
import os
import pickle
import math
import shutil
import numpy as np
import torch
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
from types import SimpleNamespace

from rare_const import (
    PromptType,
    DEFAULT_MODEL,
    LANGUAGE,
    MAX_COST_USD,
    MAX_LOOP_COUNT,
    MAX_API_CALLS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TARGET_COUNT,
    EXCLUDE_KEYWORDS,
    VALIDITY_CONFIDENCE_THRESHOLD,
)
from rare_entities import (
    AtomicInfo,
    RedundancyMapping,
    RareEvaluationItem,
    RedundancyPipelineResult,
    CostSummary,
    STEP1_SETTINGS,
    STEP2_SETTINGS,
    STEP3_SETTINGS,
    STEP4_SETTINGS,
    STEP5_SETTINGS,
    STEP6_SETTINGS,
    STEP7_SETTINGS,
)
from rare_core.rare_llm_client_service import LLMClient
from rare_core.rare_search_client_service import SearchClient
from rare_core.rare_json_parser_service import clean_and_parse_json
from rare_core.rare_prompt_maker_service import generate_prompt
from rare_core.rare_ranking_utils import (
    generate_ranking_from_scores, comprehensive_ranking_comparison, 
    generate_chunk_based_rankings, generate_chunk_based_rankings_by_rank_average
)
from rare_core.rare_document_processor import DocumentProcessor




logger = logging.getLogger(__name__)


STEP7_LOGICAL_TYPE_MAPPING = {
    "contextual_independence": "context_assumption_errors",
    "answer_exclusion": "circular_definition_errors",
    "information_equivalence": "information_equivalence_errors",
    "question_ambiguity": "question_ambiguity_errors",
}


@dataclass
class _Step7AtomicInfoData:
    atomic_id: str
    content: str
    chunk_id: str


@dataclass
class _Step7QuestionSample:
    sample_id: str
    question: str
    answer: str
    connectivity_score: float
    fluency_score: float
    essentiality_score: float
    validity_score: float
    rrf_score: float
    atomic_info_list: List[_Step7AtomicInfoData]
    gold_chunks: List[List[str]]
    generation_reasoning: str
    num_information_used: int
    context_independence_reasoning: str = ""
    circular_definition_reasoning: str = ""
    information_completeness_reasoning: str = ""
    information_equivalence_reasoning: str = ""
    question_ambiguity_reasoning: str = ""
    answerability_reasoning: str = ""


class _Step7ValidationTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.progress = {
            "generation": 0,
            "consistency": 0,
            "answerability": 0,
            "validation": 0,
        }
        self.failed_samples: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            "contextual_independence": {"passed": 0, "failed": 0, "total": 0},
            "answer_exclusion": {"passed": 0, "failed": 0, "total": 0},
            "information_equivalence": {"passed": 0, "failed": 0, "total": 0},
            "question_ambiguity": {"passed": 0, "failed": 0, "total": 0},
            "answerability": {"passed": 0, "failed": 0, "total": 0},
        }

    def increment_progress(self, step: str):
        with self.lock:
            if step in self.progress:
                self.progress[step] += 1

    def record_logical_failure(
        self,
        sample_id: str,
        question: str,
        failure_type: str,
        reasoning: str,
        selected_items: List[int],
        question_id: str = "",
        validation_details: Optional[Dict[str, Any]] = None,
        selected_contents: Optional[List[Dict[str, Any]]] = None,
        answer: str = "",
    ):
        with self.lock:
            question_key = f"{sample_id}_{hash(question) % 10000:04d}"
            if question_key not in self.failed_samples:
                self.failed_samples[question_key] = {
                    "sample_id": sample_id,
                    "question": question,
                    "answer": answer,
                    "failure_types": [],
                    "failure_details": {},
                    "selected_items": selected_contents or [],
                    "question_id": question_id,
                }

            record = self.failed_samples[question_key]
            if failure_type not in record["failure_types"]:
                record["failure_types"].append(failure_type)
                record["failure_details"][failure_type] = {"reasoning": reasoning}
                if validation_details:
                    record["failure_details"][failure_type]["validation_details"] = validation_details

            self.stats[failure_type]["failed"] += 1
            self.stats[failure_type]["total"] += 1

    def record_logical_success(self, failure_type: str):
        with self.lock:
            self.stats[failure_type]["passed"] += 1
            self.stats[failure_type]["total"] += 1

    def record_answerability_failure(
        self,
        sample_id: str,
        question: str,
        selected_items: List[int],
        chunk_ids: List[str],
        reasoning: str = "",
        selected_contents: Optional[List[Dict[str, Any]]] = None,
        answer: str = "",
    ):
        with self.lock:
            question_key = f"{sample_id}_{hash(question) % 10000:04d}"
            if question_key not in self.failed_samples:
                self.failed_samples[question_key] = {
                    "sample_id": sample_id,
                    "question": question,
                    "answer": answer,
                    "failure_types": [],
                    "failure_details": {},
                    "selected_items": selected_contents or [],
                    "question_id": "",
                }

            record = self.failed_samples[question_key]
            if "answerability" not in record["failure_types"]:
                record["failure_types"].append("answerability")
                record["failure_details"]["answerability"] = {
                    "reasoning": reasoning or "Question is not answerable with provided chunks",
                    "chunk_ids": chunk_ids,
                }

            self.stats["answerability"]["failed"] += 1
            self.stats["answerability"]["total"] += 1

    def record_answerability_success(self):
        with self.lock:
            self.stats["answerability"]["passed"] += 1
            self.stats["answerability"]["total"] += 1


def run_rare_pipeline(
    input_path: Optional[str] = None,
    all_docs: Optional[List[dict]] = None,
    target_count: int = DEFAULT_TARGET_COUNT,
    model: str = DEFAULT_MODEL,
    language: str = LANGUAGE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    api_key: str = None,
    output_dir: str = "rare_output",
    steps: List[str] = None,
    chunk_size: int = STEP2_SETTINGS.chunk_size,
    chunk_overlap: int = STEP2_SETTINGS.overlap,
    max_workers: int = STEP3_SETTINGS.workers,
    top_k_per_document: int = None,
    top_k_per_chunk: int = STEP6_SETTINGS.top_k_per_chunk,
    step7_generation_model: str = STEP7_SETTINGS.generation_model,
    step7_filter_model: str = STEP7_SETTINGS.filter_model,
    step7_validation_model: str = STEP7_SETTINGS.validation_model,
    # step7_paraphrase_model: str = STEP7_SETTINGS.paraphrase_model,
    step7_answerability_model: str = STEP7_SETTINGS.answerability_model,
    step7_num_information: int = STEP7_SETTINGS.num_information,
    step7_num_questions: int = STEP7_SETTINGS.num_questions,
    step7_num_samples: Optional[int] = None,
    step7_input_pool_size: int = STEP7_SETTINGS.input_pool_size,
    step7_max_workers: int = STEP7_SETTINGS.max_workers,
    step7_legacy_mode: bool = STEP7_SETTINGS.legacy_mode,
) -> RedundancyPipelineResult:
    """Run RARE pipeline with redundancy-aware evaluation item generation"""
    
    # Default steps if not specified
    if steps is None:
        steps = ["parsing", "chunking", "atomic_info", "selection", "embedding", "redundancy", "evaluation"]
    
    # Initialize orchestrator
    try:
        similarity_threshold_value = float(similarity_threshold)
    except (TypeError, ValueError):
        similarity_threshold_value = float(DEFAULT_SIMILARITY_THRESHOLD)

    orchestrator = RareOrchestrator(
        api_key=api_key,
        similarity_threshold=similarity_threshold_value,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    logger.info(f"Pipeline starting - Steps: {steps}")
    
    try:
        result_data = {}
        
        # Step 1: Document Parsing (PDF -> Text)
        if "parsing" in steps:
            if not input_path:
                raise ValueError("input_path required for document parsing step")
            result_data["parsed_texts"] = orchestrator.step1_document_parsing(input_path)
            logger.info(f"Step 1: Completed - {len(result_data['parsed_texts'])} documents")
        
        # Step 2: Document Chunking (Text -> Chunks)  
        if "chunking" in steps:
            if "parsed_texts" not in result_data:
                result_data["parsed_texts"] = orchestrator.load_step_result("step1_parsed_texts")
            result_data["chunks"] = orchestrator.step2_document_chunking(result_data["parsed_texts"])
            logger.info(f"Step 2: Completed - {len(result_data['chunks'])} chunks")
        
        # Step 3: Atomic Information Extraction
        if "atomic_info_extraction" in steps:
            if "chunks" not in result_data:
                if all_docs:
                    result_data["chunks"] = all_docs
                else:
                    result_data["chunks"] = orchestrator.load_step_result("step2_chunks")
            result_data["atomic_info_map"] = orchestrator.step3_atomic_info_extraction(
                result_data["chunks"], language, model, max_workers
            )
            total_atomic = sum(len(units) for units in result_data["atomic_info_map"].values())
            logger.info(f"Step 3: Completed - {total_atomic} atomic info")
        
        # Step 4: Best Information Selection (Separate 5-prompt + RRF Ranking)
        if "atomic_info_selection" in steps:
            if "atomic_info_map" not in result_data:
                result_data["atomic_info_map"] = orchestrator.load_step_result("step3_atomic_info_map")
            if "chunks" not in result_data:
                result_data["chunks"] = orchestrator.load_step_result("step2_chunks")
            result_data["selected_atomic_info_map"] = orchestrator.step4_best_info_selection(
                atomic_info_map=result_data["atomic_info_map"],
                chunks=result_data["chunks"],
                language=language,
                model=model,
                max_workers=STEP4_SETTINGS.workers,
                enable_logical_filtering=not STEP4_SETTINGS.skip_logical_filtering,
                logical_filtering_model=STEP4_SETTINGS.logical_filtering_model,
            )
            # Extract count from the structured result
            threshold_data = result_data["selected_atomic_info_map"]["comparison_modes"]["threshold_filtered"]["atomic_info_by_chunk"]
            total_selected = sum(len(units) for units in threshold_data.values())
            total_original = sum(len(units) for units in result_data["atomic_info_map"].values())
            filtered_count = total_original - total_selected
            logger.info(f"Step 4: Completed - {total_selected} selected")
        
        # Step 5: Build Embeddings
        if "embedding_similarity" in steps:
            if "selected_atomic_info_map" in result_data:
                atomic_info_for_embedding = {}
                threshold_data = result_data["selected_atomic_info_map"]["comparison_modes"]["threshold_filtered"]["atomic_info_by_chunk"]
                for chunk_id, atomic_list in threshold_data.items():
                    atomic_info_for_embedding[chunk_id] = [
                        AtomicInfo(
                            content=ai["content"],
                            chunk_id=ai["chunk_id"],
                            atomic_info_id=ai["atomic_info_id"]
                        ) for ai in atomic_list
                    ]
                result_data["embeddings"] = orchestrator.step5_build_embeddings(
                    atomic_info_for_embedding,
                    similarity_threshold=STEP5_SETTINGS.similarity_threshold,
                    auto_threshold_batch_size=STEP5_SETTINGS.auto_threshold_batch_size,
                    embedding_batch_size=STEP5_SETTINGS.batch_size,
                    similarity_only=STEP5_SETTINGS.similarity_only,
                    embedding_only=STEP5_SETTINGS.embedding_only,
                    similarity_batch_size=STEP5_SETTINGS.similarity_batch_size,
                )
            else:
                if "atomic_info_map" not in result_data:
                    result_data["atomic_info_map"] = orchestrator.load_step_result("step3_atomic_info_map")

                result_data["embeddings"] = orchestrator.step5_build_embeddings(
                    result_data["atomic_info_map"],
                    similarity_threshold=STEP5_SETTINGS.similarity_threshold,
                    auto_threshold_batch_size=STEP5_SETTINGS.auto_threshold_batch_size,
                    embedding_batch_size=STEP5_SETTINGS.batch_size,
                    similarity_only=STEP5_SETTINGS.similarity_only,
                    embedding_only=STEP5_SETTINGS.embedding_only,
                    similarity_batch_size=STEP5_SETTINGS.similarity_batch_size,
                )
        logger.info(f"Step 5: Embeddings completed")
        
        # Step 6: Redundancy Detection
        if "redundancy_detection" in steps:
            # Use selected atomic info if available, otherwise use raw atomic info
            if "selected_atomic_info_map" in result_data:
                atomic_info_for_redundancy = {}
                threshold_data = result_data["selected_atomic_info_map"]["comparison_modes"]["threshold_filtered"]["atomic_info_by_chunk"]
                for chunk_id, atomic_list in threshold_data.items():
                    atomic_info_for_redundancy[chunk_id] = [
                        AtomicInfo(
                            content=ai["content"],
                            chunk_id=ai["chunk_id"], 
                            atomic_info_id=ai["atomic_info_id"]
                        ) for ai in atomic_list
                    ]
            else:
                if "atomic_info_map" not in result_data:
                    result_data["atomic_info_map"] = orchestrator.load_step_result("step3_atomic_info_map")
                atomic_info_for_redundancy = result_data["atomic_info_map"]
            
            if "embeddings" not in result_data:
                result_data["embeddings"] = orchestrator.load_step_result("step5_embeddings")
            result_data["redundancy_mapping"] = orchestrator.step6_redundancy_detection(
                atomic_info_map=atomic_info_for_redundancy,
                embeddings_data=result_data["embeddings"],
                language=language,
                model=model,
                max_workers=STEP6_SETTINGS.max_workers,
                max_similar_items=STEP6_SETTINGS.max_similar_items,
                top_k_per_chunk=top_k_per_chunk,
                max_chunks=STEP6_SETTINGS.max_chunks,
            )
            unique_count = sum(1 for r in result_data["redundancy_mapping"].values() if r.redundant_items == ["unique"])
            total_count = len(result_data["redundancy_mapping"])
            logger.info(f"Step 6: Completed - {unique_count} unique items")
        
        # Step 7: Evaluation Generation (optional)
        run_step7 = "data_generation" in steps
        if run_step7:
            if "redundancy_mapping" not in result_data:
                result_data["redundancy_mapping"] = orchestrator.load_step_result("step6_redundancy_mapping")
            if "chunks" not in result_data:
                result_data["chunks"] = orchestrator.load_step_result("step2_chunks")

            redundancy_mapping = result_data.get("redundancy_mapping")
            chunk_source = result_data.get("chunks")

            if redundancy_mapping and chunk_source:
                chunk_data = orchestrator._convert_chunks_to_dict(chunk_source)
                effective_num_samples = step7_num_samples if step7_num_samples is not None else target_count
                step7_result = orchestrator.step7_multihop_question_generation(
                    redundancy_mapping=redundancy_mapping,
                    chunk_data=chunk_data,
                    num_information=step7_num_information,
                    num_sample=effective_num_samples,
                    num_questions=step7_num_questions,
                    input_pool_size=step7_input_pool_size,
                    generation_model=step7_generation_model,
                    filter_model=step7_filter_model,
                    validation_model=step7_validation_model,
                    answerability_model=step7_answerability_model,
                    language=language,
                    max_workers=step7_max_workers,
                    legacy_mode=step7_legacy_mode,
                )

                result_data["multihop_questions"] = [asdict(sample) for sample in step7_result["generated_samples"]]
                logger.info(
                    "Step 7: Completed - %d questions",
                    len(result_data["multihop_questions"]),
                )
        
        # Create final result
        final_result = RedundancyPipelineResult(
            atomic_info_map=result_data.get("atomic_info_map", {}),
            redundancy_mapping=result_data.get("redundancy_mapping", {}),
            evaluation_items=result_data.get("multihop_questions", []),
            cost_summary=orchestrator.llm_client.get_cost_summary(),
            statistics=orchestrator.get_pipeline_statistics(
                result_data.get("atomic_info_map", {}),
                result_data.get("redundancy_mapping", {}), 
                result_data.get("multihop_questions", [])
            )
        )
        
        # Show cumulative cost
        cost = final_result.cost_summary
        logger.info(f"Pipeline completed - Cost: ${cost.total_cost_usd:.6f}")
        
        return final_result
        
    except Exception as e:
        logger.error(f"[RARE] Pipeline failed: {str(e)}")
        raise


class RareOrchestrator:
    """Main orchestrator for RARE redundancy-aware pipeline"""
    
    def __init__(
        self,
        api_key: str = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        output_dir: str = "rare_output",
        chunk_size: int = STEP2_SETTINGS.chunk_size,
        chunk_overlap: int = STEP2_SETTINGS.overlap,
    ):
        self.llm_client = LLMClient(api_key=api_key)
        self.similarity_threshold = similarity_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Document processor for parsing and chunking
        self.doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # =============================================================================
    # Step-by-Step Pipeline Functions
    # =============================================================================
    
    def step1_document_parsing(self, input_path: str) -> Dict[str, Any]:
        """Step 1: Parse PDF documents to text"""
        
        logger.info(f"Step 1: Document parsing")
        
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file() and input_path_obj.suffix.lower() == '.pdf':
            # Single PDF file
            page_texts = self.doc_processor.extract_text_from_pdf(str(input_path_obj))
            parsed_data = {
                str(input_path_obj.name): page_texts
            }
        elif input_path_obj.is_dir():
            # Folder of PDF files
            parsed_data = {}
            pdf_files = list(input_path_obj.glob("*.pdf"))
            
            with tqdm(pdf_files, desc="Parsing PDFs", unit="files") as pbar:
                for pdf_file in pbar:
                    try:
                        page_texts = self.doc_processor.extract_text_from_pdf(str(pdf_file))
                        parsed_data[pdf_file.name] = page_texts
                        
                        pbar.set_postfix({
                            'Current': pdf_file.name[:20] + "...",
                            'Pages': len(page_texts)
                        })
                        
                    except Exception as e:
                        logger.error(f"[Step 1] Failed to parse {pdf_file.name}: {e}")
                        pbar.set_postfix({'Error': pdf_file.name})
        else:
            raise ValueError(f"Input path must be PDF file or directory: {input_path}")
        
        # Save result
        output_file = self.output_dir / "step1_parsed_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Step 1: Completed - {len(parsed_data)} documents")
        
        return parsed_data
    
    def step2_document_chunking(self, parsed_texts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Step 2: Chunk parsed texts into smaller pieces"""
        
        logger.info(f"Step 2: Document chunking")
        
        all_chunks = []
        total_pages = sum(len(page_texts) for page_texts in parsed_texts.values())
        
        with tqdm(total=total_pages, desc="Chunking text", unit="pages") as pbar:
            for file_name, page_texts in parsed_texts.items():
                for page_no, text in page_texts.items():
                    if isinstance(page_no, str):
                        page_no = int(page_no)
                    
                    chunks = self.doc_processor.chunk_text(text, file_name, page_no)
                    all_chunks.extend(chunks)
                    
                    pbar.set_postfix({
                        'File': file_name[:15] + "...",
                        'Page': page_no,
                        'Chunks': len(chunks)
                    })
                    pbar.update(1)
        
        # Save result
        output_file = self.output_dir / "step2_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Step 2: Completed - {len(all_chunks)} chunks")
        
        return all_chunks
    
    def step3_atomic_info_extraction(
        self, 
        chunks: List[dict], 
        language: str, 
        model: str,
        max_workers: int = 64
    ) -> Dict[str, List[AtomicInfo]]:
        """Step 3: Extract atomic information from chunks using multithreading"""
        
        logger.info(f"Step 3: Atomic info extraction - {len(chunks)} chunks")
        
        atomic_info_map = {}
        
        # Prepare task list
        tasks = []
        for i, chunk in enumerate(chunks):
            doc_id = self._generate_doc_id(chunk, i)
            content = chunk.get("content", "").strip()
            doc_title = chunk.get("file_name", "")
            
            if content:
                tasks.append((content, doc_id, language, model, doc_title, i))
        
        # Process in parallel using ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._extract_atomic_info_single, content, doc_id, language, model, doc_title): 
                (doc_id, idx) for content, doc_id, language, model, doc_title, idx in tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(tasks), desc="Extracting atomic info", unit="chunks") as pbar:
                for future in as_completed(future_to_task):
                    doc_id, task_idx = future_to_task[future]
                    
                    try:
                        atomic_info_list = future.result()
                        if atomic_info_list:
                            atomic_info_map[doc_id] = atomic_info_list
                        
                    except Exception as e:
                        logger.error(f"[Step 3] Error processing chunk {task_idx+1}: {e}")
                    
                    # Update progress bar (use consistent cost method like Step 4)
                    current_cost = self.llm_client.get_cost_summary().total_cost_usd
                    pbar.set_postfix({
                        'Cost': f'${current_cost:.4f}',
                        'Processed': f'{pbar.n+1}/{len(tasks)}'
                    })
                    pbar.update(1)
        
        # Save result (convert AtomicInfo to dict for JSON serialization)
        atomic_info_dict = {}
        for chunk_id, atomic_list in atomic_info_map.items():
            atomic_info_dict[chunk_id] = [
                {
                    "content": ai.content,
                    "chunk_id": ai.chunk_id,
                    "atomic_info_id": ai.atomic_info_id
                }
                for ai in atomic_list
            ]
        
        output_file = self.output_dir / "step3_atomic_info_map.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(atomic_info_dict, f, ensure_ascii=False, indent=2)
        
        total_atomic = sum(len(units) for units in atomic_info_map.values())
        logger.info(f"Step 3: Completed - {total_atomic} atomic units")
        
        return atomic_info_map
    
    def step3_5_validate_atomic_info(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        chunks: List[dict],
        language: str,
        model: str,
        max_workers: int = 64,
        fewshot_examples_by_chunk: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[AtomicInfo]]:
        """Step 3.5: Validate atomic information quality using MIRA's validation approach"""
        
        total_input_chunks = len(atomic_info_map)
        total_atomic_units = sum(len(units) for units in atomic_info_map.values())
        logger.info(f"Step 3.5: Validation - {total_atomic_units} units")
        
        validated_atomic_info_map = {}
        
        # Create chunk lookup for document content
        chunk_lookup = {}
        for chunk in chunks:
            doc_id = self._generate_doc_id(chunk, 0)  # We'll match by file/content
            chunk_lookup[chunk.get("file_name", "unknown")] = chunk.get("content", "")
        
        # Group atomic info by chunk for bulk validation
        chunk_validation_tasks = {}
        total_filtered_count = 0
        
        for chunk_id, atomic_list in atomic_info_map.items():
            # Get document content and title for this chunk
            file_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            doc_content = chunk_lookup.get(file_name, "")
            doc_title = file_name  # Use actual file name as title
            
            # Use all atomic info (keyword filtering removed)
            filtered_atomic_list = atomic_list
            
            if filtered_atomic_list:  # Only add if there are atomic info to validate
                chunk_validation_tasks[chunk_id] = (filtered_atomic_list, doc_content, doc_title, language, model)
        
        logger.info(f"Step 3.5: After filtering - {total_filtered_count} items")
        logger.info(f"[DEBUG Combined] Input chunks: {total_input_chunks}, Validation tasks: {len(chunk_validation_tasks)}")
        
        # Process bulk validation with progress bar
        valid_atomic_info = []  # Threshold-filtered atomic info
        all_atomic_info = []    # All atomic info (for dual comparison mode)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit bulk validation tasks (one per chunk)
            future_to_chunk = {
                executor.submit(
                    self._validate_bulk_atomic_info,
                    atomic_list,
                    doc_content,
                    doc_title,
                    language,
                    model,
                    (fewshot_examples_by_chunk.get(chunk_id) if fewshot_examples_by_chunk else None)
                ): chunk_id
                for chunk_id, (atomic_list, doc_content, doc_title, language, model) in chunk_validation_tasks.items()
            }
            
            # Process results with progress bar
            with tqdm(total=len(chunk_validation_tasks), desc="Combined validating chunks", unit="chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    
                    try:
                        validated_atomic_list = future.result()  # Returns List[AtomicInfo] with updated scores
                        
                        # DEBUG: Track chunk processing
                        logger.info(f"[DEBUG Combined] Chunk {chunk_id}: received {len(validated_atomic_list)} atomic items")
                        
                        # Count scores by availability
                        zero_confidence_count = 0
                        valid_threshold_count = 0
                        
                        # Collect ALL atomic info (for dual comparison mode) - EXCLUDE zero-scored items
                        for validated_atomic_info in validated_atomic_list:
                            # Only include items that were actually validated (non-zero confidence)
                            if validated_atomic_info.overall_confidence > 0.0:
                                all_atomic_info.append((validated_atomic_info, chunk_id))
                            else:
                                zero_confidence_count += 1
                        
                        # Filter valid atomic info based on overall_confidence threshold
                        for validated_atomic_info in validated_atomic_list:
                            # logger.debug(f"[DEBUG COMBINED] Item: '{validated_atomic_info.content[:30]}...' - confidence: {validated_atomic_info.overall_confidence:.3f}, threshold: {VALIDITY_CONFIDENCE_THRESHOLD}")
                            
                            if validated_atomic_info.overall_confidence >= VALIDITY_CONFIDENCE_THRESHOLD:
                                valid_atomic_info.append((validated_atomic_info, chunk_id))
                                valid_threshold_count += 1
                                logger.debug(f"[Step 3.5] Valid: {validated_atomic_info.content[:50]}... " +
                                           f"(overall: {validated_atomic_info.overall_confidence:.2f}, " +
                                           f"v: {validated_atomic_info.validity_score:.2f}, " +
                                           f"c: {validated_atomic_info.completeness_score:.2f}, " +
                                           f"s: {validated_atomic_info.specificity_score:.2f})", 
                                           f"cl: {validated_atomic_info.clarity_score:.2f}, " +
                                           f"q: {validated_atomic_info.questionability_score:.2f})"
                                           ) 
                            else:
                                logger.debug(f"[Step 3.5] Invalid: {validated_atomic_info.content[:50]}... " +
                                           f"(overall: {validated_atomic_info.overall_confidence:.2f})")
                        
                        # DEBUG: Chunk summary
                        logger.info(f"[DEBUG Combined] Chunk {chunk_id}: {zero_confidence_count} zero-confidence, {valid_threshold_count} above threshold")
                        
                    except Exception as e:
                        logger.error(f"[Step 3.5] Error combined validating chunk {chunk_id}: {e}")
                        logger.info(f"[DEBUG Combined] Chunk {chunk_id}: FAILED validation")
                    
                    # Update progress bar with detailed metrics
                    current_cost = self.llm_client.get_cost_summary().total_cost_usd
                    avg_overall = sum(item[0].overall_confidence for item in valid_atomic_info) / len(valid_atomic_info) if valid_atomic_info else 0.0
                    processed_chunks = pbar.n + 1
                    pbar.set_postfix({
                        'Cost': f'${current_cost:.4f}',
                        'Valid_Items': len(valid_atomic_info),
                        'Avg_Score': f'{avg_overall:.2f}',
                        'Chunks': f'{processed_chunks}/{len(chunk_validation_tasks)}'
                    })
                    pbar.update(1)
        
        # DEBUG: Final summary statistics for Combined validation
        logger.info(f"[DEBUG Combined] FINAL SUMMARY:")
        logger.info(f"[DEBUG Combined] Input chunks: {total_input_chunks}")
        logger.info(f"[DEBUG Combined] Validation tasks: {len(chunk_validation_tasks)}")
        logger.info(f"[DEBUG Combined] All atomic items (confidence > 0): {len(all_atomic_info)}")
        logger.info(f"[DEBUG Combined] Valid atomic items (>= {VALIDITY_CONFIDENCE_THRESHOLD}): {len(valid_atomic_info)}")
        
        # Group validated atomic info back by chunk_id (threshold filtered)
        validated_atomic_info_map = {}
        for atomic_info, chunk_id in valid_atomic_info:
            if chunk_id not in validated_atomic_info_map:
                validated_atomic_info_map[chunk_id] = []
            validated_atomic_info_map[chunk_id].append(atomic_info)
        
        # Group ALL atomic info back by chunk_id (for dual comparison)
        all_atomic_info_map = {}
        for atomic_info, chunk_id in all_atomic_info:
            if chunk_id not in all_atomic_info_map:
                all_atomic_info_map[chunk_id] = []
            all_atomic_info_map[chunk_id].append(atomic_info)
        
        # DEBUG: Final chunk count after grouping
        logger.info(f"[DEBUG Combined] FINAL OUTPUT CHUNKS:")
        logger.info(f"[DEBUG Combined] Threshold-filtered chunks: {len(validated_atomic_info_map)}")
        logger.info(f"[DEBUG Combined] All chunks (confidence > 0): {len(all_atomic_info_map)}")
        
        # Save validated result WITH ALL validation scores
        validated_dict = {}
        for chunk_id, atomic_list in validated_atomic_info_map.items():
            validated_dict[chunk_id] = [
                {
                    "content": ai.content,
                    "chunk_id": ai.chunk_id,
                    "atomic_info_id": ai.atomic_info_id,
                    "validity_score": ai.validity_score,
                    "completeness_score": ai.completeness_score,
                    "specificity_score": ai.specificity_score,
                    "clarity_score": ai.clarity_score,
                    "questionability_score": ai.questionability_score,
                    "overall_confidence": ai.overall_confidence,
                    "validation_reasoning": ai.validation_reasoning
                }
                for ai in atomic_list
            ]
        
        # Calculate and add chunk-based rankings for both threshold-filtered and all data
        # 1. Threshold-filtered rankings
        flat_atomic_list = []
        for atomic_list in validated_atomic_info_map.values():
            flat_atomic_list.extend(atomic_list)
        
        chunk_rankings_filtered = generate_chunk_based_rankings_by_rank_average(flat_atomic_list)
        
        # 2. All data rankings (for dual comparison mode)
        all_flat_atomic_list = []
        for atomic_list in all_atomic_info_map.values():
            all_flat_atomic_list.extend(atomic_list)
        
        chunk_rankings_all = generate_chunk_based_rankings_by_rank_average(all_flat_atomic_list)
        
        # Save all atomic info data structure (for dual comparison)
        all_atomic_dict = {}
        for chunk_id, atomic_list in all_atomic_info_map.items():
            all_atomic_dict[chunk_id] = [
                {
                    "content": ai.content,
                    "chunk_id": ai.chunk_id,
                    "atomic_info_id": ai.atomic_info_id,
                    "validity_score": ai.validity_score,
                    "completeness_score": ai.completeness_score,
                    "specificity_score": ai.specificity_score,
                    "overall_confidence": ai.overall_confidence,
                    "validation_reasoning": ai.validation_reasoning
                }
                for ai in atomic_list
            ]
        
        # Create simple ranking by chunk (like GOLD ranking) - sorted by chunk ID
        validation_ranking_by_chunk = {}
        for chunk_id in sorted(chunk_rankings_all.keys()):
            validation_ranking_by_chunk[chunk_id] = chunk_rankings_all[chunk_id]
        
        # Add ranking information to save data (dual comparison mode)
        validated_dict_with_rankings = {
            "validation_method": "combined",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "validation_ranking_by_chunk": validation_ranking_by_chunk,  # Simple ranking like GOLD
            "comparison_modes": {
                "threshold_filtered": {
                    "atomic_info_by_chunk": validated_dict,
                    "chunk_rankings": chunk_rankings_filtered,
                    "threshold": VALIDITY_CONFIDENCE_THRESHOLD,
                    "total_items": len(flat_atomic_list)
                },
                "all_items": {
                    "atomic_info_by_chunk": all_atomic_dict,
                    "chunk_rankings": chunk_rankings_all,
                    "threshold": "none",
                    "total_items": len(all_flat_atomic_list)
                }
            },
            "ranking_metadata": {
                "ranking_method": "overall_confidence_desc",
                "total_chunks": len(chunk_rankings_all)
            }
        }
        
        output_file = self.output_dir / "step3_5_validated_atomic_info_combined.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_dict_with_rankings, f, ensure_ascii=False, indent=2)
        
        total_validated = sum(len(units) for units in validated_atomic_info_map.values())
        logger.info(f"Step 3.5: Completed - {total_validated} validated")
        
        # Return both filtered and all data for SQuAD evaluation compatibility
        return validated_dict_with_rankings

    def step4_best_info_selection(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        chunks: List[dict],
        language: str,
        model: str,
        max_workers: int = 128,
        enable_logical_filtering: bool = True,
        logical_filtering_model: str = "gpt5_nano"
    ) -> Dict[str, List[AtomicInfo]]:
        """Step 4: Validate atomic information with optional logical consistency pre-filtering + 5 separate LLM calls with RRF ranking"""
        
        total_atomic_count = sum(len(units) for units in atomic_info_map.values())
        
        if enable_logical_filtering:
            logger.info(f"Step 4: Processing - {total_atomic_count} units (completeness+RRF)")
        else:
            logger.info(f"Step 4: Processing - {total_atomic_count} units (separate+RRF)")
        
        # Step 4a: Optional Completeness Pre-filtering
        if enable_logical_filtering:
            logger.info("Step 4a: Completeness filtering")
            
            # Flatten all atomic info for logical filtering
            all_atomic_info = []
            atomic_info_chunk_mapping = {}  # Track which chunk each atomic info belongs to
            
            for chunk_id, atomic_list in atomic_info_map.items():
                for atomic_info in atomic_list:
                    all_atomic_info.append(atomic_info)
                    atomic_info_chunk_mapping[atomic_info.atomic_info_id] = chunk_id
            
            # Apply information completeness filtering
            filtering_result = self.filter_atomic_info_logical_consistency(
                atomic_info_list=all_atomic_info,
                model=logical_filtering_model,
                max_workers=max_workers
            )
            
            valid_atomic_info = filtering_result['valid_atomic_info']
            filtering_stats = filtering_result['detailed_stats']
            
            # Rebuild atomic_info_map with only completeness-valid atomic info
            filtered_atomic_info_map = {}
            for atomic_info in valid_atomic_info:
                chunk_id = atomic_info_chunk_mapping[atomic_info.atomic_info_id]
                if chunk_id not in filtered_atomic_info_map:
                    filtered_atomic_info_map[chunk_id] = []
                filtered_atomic_info_map[chunk_id].append(atomic_info)
            
            # Update atomic_info_map for validation step
            atomic_info_map = filtered_atomic_info_map
            
            total_after_filtering = sum(len(units) for units in atomic_info_map.values())
            filtering_rate = (total_after_filtering / total_atomic_count * 100) if total_atomic_count > 0 else 0
            
            logger.info(f"Step 4a: Completed - {total_after_filtering}/{total_atomic_count} passed"
                       f"Incomplete:{filtering_stats['information_completeness_errors']}")
            
            # Print user-friendly filtering results
            print(f"Completeness Filtering Results: {total_after_filtering}/{total_atomic_count} atomic info passed ({filtering_rate:.1f}%)")
            print(f"Filtered out: Incomplete={filtering_stats['information_completeness_errors']}")
            print()
        
        # Step 4b: Validation and RRF Ranking (existing logic)
        logger.info("Step 4b: Validation and ranking")
        validated_atomic_info_map = {}
        
        # Create chunk lookup for document content
        chunk_lookup = {}
        for chunk in chunks:
            chunk_lookup[chunk.get("file_name", "unknown")] = chunk.get("content", "")
        
        # Process chunks in parallel (like Combined validation, but with 5 calls per chunk)
        chunk_validation_tasks = []
        total_filtered_count = 0
        
        for chunk_id, atomic_list in atomic_info_map.items():
            # Get document content and title for this chunk
            file_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            doc_content = chunk_lookup.get(file_name, "")
            doc_title = file_name  # Use actual file name as title
            
            # Use all atomic info (keyword filtering removed, logical filtering already applied if enabled)
            filtered_atomic_list = atomic_list
            
            if filtered_atomic_list:
                chunk_validation_tasks.append((chunk_id, filtered_atomic_list, doc_content, doc_title, language, model))
                total_filtered_count += len(filtered_atomic_list)
        
        logger.info(f"Step 4: After filtering - {total_filtered_count} items")
        
        # Process validation chunk by chunk (3 calls per chunk)
        valid_atomic_info = []  # For threshold filtering (>= 0.7)
        all_atomic_info = []    # For dual comparison mode (> 0.0)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunk validation tasks 
            future_to_task = {
                executor.submit(self._validate_separate_chunk_atomic_info, atomic_list, doc_content, doc_title, language, model): 
                chunk_id for chunk_id, atomic_list, doc_content, doc_title, language, model in chunk_validation_tasks
            }
            
            # Process results with progress bar
            with tqdm(total=len(chunk_validation_tasks), desc="RRF validating chunks", unit="chunks") as pbar:
                for future in as_completed(future_to_task):
                    chunk_id = future_to_task[future]
                    
                    try:
                        validated_chunk_atomic_list = future.result()  # Returns list of validated AtomicInfo objects
                        
                        for validated_atomic_info in validated_chunk_atomic_list:
                            # Collect ALL atomic info (for dual comparison mode) - EXCLUDE zero-scored items
                            if validated_atomic_info.overall_confidence > 0.0:
                                all_atomic_info.append((validated_atomic_info, chunk_id))
                            
                            # logger.debug(f"[DEBUG SEPARATE] Item: '{validated_atomic_info.content[:30]}...' - confidence: {validated_atomic_info.overall_confidence:.3f}, threshold: {VALIDITY_CONFIDENCE_THRESHOLD}")
                            
                            # Check if valid based on overall_confidence threshold
                            if validated_atomic_info.overall_confidence >= VALIDITY_CONFIDENCE_THRESHOLD:
                                valid_atomic_info.append((validated_atomic_info, chunk_id))
                                logger.debug(f"[Step 3.5 SEPARATE] Valid: {validated_atomic_info.content[:50]}... " +
                                           f"(overall: {validated_atomic_info.overall_confidence:.2f}, " +
                                           f"v: {validated_atomic_info.validity_score:.2f}, " +
                                           f"c: {validated_atomic_info.completeness_score:.2f}, " +
                                           f"s: {validated_atomic_info.specificity_score:.2f}, " +
                                           f"cl: {validated_atomic_info.clarity_score:.2f}, " +
                                           f"q: {validated_atomic_info.questionability_score:.2f})"
                                           )
                            else:
                                logger.debug(f"[Step 3.5 SEPARATE] Invalid: {validated_atomic_info.content[:50]}... " +
                                           f"(overall: {validated_atomic_info.overall_confidence:.2f})")
                            
                    except Exception as e:
                        logger.error(f"[Step 3.5 SEPARATE] Error validating chunk {chunk_id}: {e}")
                    
                    # Update progress bar with detailed metrics  
                    current_cost = self.llm_client.get_cost_summary().total_cost_usd
                    avg_overall = sum(item[0].overall_confidence for item in valid_atomic_info) / len(valid_atomic_info) if valid_atomic_info else 0.0
                    pbar.set_postfix({
                        'Cost': f'${current_cost:.4f}',
                        'Valid_Items': len(valid_atomic_info),
                        'Avg_Score': f'{avg_overall:.2f}',
                        'LLM_Calls': '3x per item'
                    })
                    pbar.update(1)
        
        # Group validated atomic info back by chunk_id (threshold filtered)
        validated_atomic_info_map = {}
        for atomic_info, chunk_id in valid_atomic_info:
            if chunk_id not in validated_atomic_info_map:
                validated_atomic_info_map[chunk_id] = []
            validated_atomic_info_map[chunk_id].append(atomic_info)
        
        # Group ALL atomic info back by chunk_id (for dual comparison mode)
        all_atomic_info_map = {}
        for atomic_info, chunk_id in all_atomic_info:
            if chunk_id not in all_atomic_info_map:
                all_atomic_info_map[chunk_id] = []
            all_atomic_info_map[chunk_id].append(atomic_info)
        
        # Save validated result WITH ALL validation scores (threshold filtered)
        validated_dict = {}
        for chunk_id, atomic_list in validated_atomic_info_map.items():
            validated_dict[chunk_id] = [
                {
                    "content": ai.content,
                    "chunk_id": ai.chunk_id,
                    "atomic_info_id": ai.atomic_info_id,
                    "validity_score": ai.validity_score,
                    "completeness_score": ai.completeness_score,
                    "specificity_score": ai.specificity_score,
                    "clarity_score": ai.clarity_score,
                    "questionability_score": ai.questionability_score,
                    "overall_confidence": ai.overall_confidence,
                    "validation_reasoning": ai.validation_reasoning
                }
                for ai in atomic_list
            ]
        
        # Calculate and add chunk-based rankings for both threshold-filtered and all data
        # 1. Threshold-filtered rankings
        flat_atomic_list = []
        for atomic_list in validated_atomic_info_map.values():
            flat_atomic_list.extend(atomic_list)
        
        chunk_rankings_filtered = generate_chunk_based_rankings_by_rank_average(flat_atomic_list)
        
        # 2. All data rankings (for dual comparison mode)
        all_flat_atomic_list = []
        for atomic_list in all_atomic_info_map.values():
            all_flat_atomic_list.extend(atomic_list)
        
        chunk_rankings_all = generate_chunk_based_rankings_by_rank_average(all_flat_atomic_list)
        
        # Save all atomic info data structure (for dual comparison)
        all_atomic_dict = {}
        for chunk_id, atomic_list in all_atomic_info_map.items():
            all_atomic_dict[chunk_id] = [
                {
                    "content": ai.content,
                    "chunk_id": ai.chunk_id,
                    "atomic_info_id": ai.atomic_info_id,
                    "validity_score": ai.validity_score,
                    "completeness_score": ai.completeness_score,
                    "specificity_score": ai.specificity_score,
                    "overall_confidence": ai.overall_confidence,
                    "validation_reasoning": ai.validation_reasoning
                }
                for ai in atomic_list
            ]
        
        # Create simple ranking by chunk (like GOLD ranking) - sorted by chunk ID
        validation_ranking_by_chunk = {}
        for chunk_id in sorted(chunk_rankings_all.keys()):
            validation_ranking_by_chunk[chunk_id] = chunk_rankings_all[chunk_id]
        
        # Add ranking information to save data (dual comparison mode)
        validated_dict_with_rankings = {
            "validation_method": "separate",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "validation_ranking_by_chunk": validation_ranking_by_chunk,  # Simple ranking like GOLD
            "comparison_modes": {
                "threshold_filtered": {
                    "atomic_info_by_chunk": validated_dict,
                    "chunk_rankings": chunk_rankings_filtered,
                    "threshold": VALIDITY_CONFIDENCE_THRESHOLD,
                    "total_items": len(flat_atomic_list)
                },
                "all_items": {
                    "atomic_info_by_chunk": all_atomic_dict,
                    "chunk_rankings": chunk_rankings_all,
                    "threshold": "none",
                    "total_items": len(all_flat_atomic_list)
                }
            },
            "ranking_metadata": {
                "ranking_method": "rrf_rank_average_k0",
                "total_chunks_threshold": len(chunk_rankings_filtered),
                "total_chunks_all": len(chunk_rankings_all),
                "validation_approach": "5_separate_llm_calls_per_chunk_with_rrf_ranking"
            }
        }
        
        output_file = self.output_dir / "step4_selected_atomic_info.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_dict_with_rankings, f, ensure_ascii=False, indent=2)
        
        total_validated = len(valid_atomic_info)
        total_llm_calls = total_validated * 3  # 3 calls per validated item
        logger.info(f"Step 3.5: Completed - {total_validated} validated")
        
        # Return both filtered and all data for SQuAD evaluation compatibility
        return validated_dict_with_rankings



    def generate_gold_ranking(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        chunks: List[dict],
        language: str = "Korean",
        model: str = "gpt-5",
        max_workers: int = 64
    ) -> List[str]:
        """Generate GOLD standard ranking using high-quality model (GPT-5)"""
        
        logger.info(f"GOLD: Generating ranking")
        
        # Flatten all atomic info from all chunks (keyword filtering removed)
        all_atomic_info = []
        for chunk_id, atomic_list in atomic_info_map.items():
            for atomic_info in atomic_list:
                all_atomic_info.append(atomic_info)
        
        if not all_atomic_info:
            logger.warning("GOLD: No atomic info available")
            return []
        
        logger.info(f"GOLD: Processing {len(all_atomic_info)} units")
        
        # Group atomic info by chunk (same as Combined validation approach)
        chunk_atomic_groups = {}
        for atomic_info in all_atomic_info:
            chunk_id = atomic_info.chunk_id
            if chunk_id not in chunk_atomic_groups:
                chunk_atomic_groups[chunk_id] = []
            chunk_atomic_groups[chunk_id].append(atomic_info)
        
        logger.info(f"GOLD: Processing {len(chunk_atomic_groups)} chunks")
        
        # Process chunks in parallel (like other validation methods)
        chunk_results = {}  # Store results by chunk_id to preserve order
        total_cost = 0.0
        total_calls = 0
        
        # Track cost before processing
        initial_cost = self.llm_client.get_cost_summary()
        
        # Prepare chunk processing tasks
        chunk_processing_tasks = []
        for chunk_id, chunk_atomic_list in chunk_atomic_groups.items():
            if not chunk_atomic_list:
                continue
            
            # Find document content for this chunk
            doc_content = ""
            doc_title = chunk_id
            for chunk in chunks:
                if chunk.get("file_name", "") == chunk_id or chunk_id in chunk.get("file_name", ""):
                    doc_content = chunk.get("content", "")[:10000]  # Limit context size
                    doc_title = chunk.get("file_name", "")
                    break
            
            chunk_processing_tasks.append((chunk_id, chunk_atomic_list, doc_content, doc_title, language, model))
        
        logger.info(f"GOLD: Processing {len(chunk_processing_tasks)} chunks parallel")
        
        # Use ThreadPoolExecutor for parallel processing (like other validation methods)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit GOLD ranking tasks (one per chunk)
            future_to_chunk = {
                executor.submit(self._generate_gold_ranking_chunk, chunk_atomic_list, doc_content, doc_title, language, model): 
                chunk_id for chunk_id, chunk_atomic_list, doc_content, doc_title, language, model in chunk_processing_tasks
            }
            
            # Process results with progress bar
            with tqdm(total=len(chunk_processing_tasks), desc="GOLD ranking chunks", unit="chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    
                    try:
                        chunk_ranking = future.result()  # Returns List[str] (atomic info content in rank order)
                        chunk_results[chunk_id] = chunk_ranking  # Store by chunk_id
                        
                    except Exception as e:
                        logger.error(f"[RARE GOLD] Error processing chunk {chunk_id}: {e}")
                        chunk_results[chunk_id] = []  # Empty ranking on error
                    
                    # Update progress bar with cumulative cost tracking
                    current_cost_usd = self.llm_client.get_cost_summary().total_cost_usd
                    current_cost_summary = self.llm_client.get_cost_summary()
                    total_cost = current_cost_summary.total_cost_usd - initial_cost.total_cost_usd
                    total_calls = current_cost_summary.total_calls - initial_cost.total_calls
                    
                    total_ranked_items = sum(len(ranking) for ranking in chunk_results.values())
                    pbar.set_postfix({
                        'Cost': f'${current_cost_usd:.4f}',
                        'Items': total_ranked_items,
                        'Chunk': f'{len(chunk_ranking) if "chunk_ranking" in locals() else 0} ranked'
                    })
                    pbar.update(1)
        
        # Show final cost
        final_cost_summary = self.llm_client.get_cost_summary()
        final_total_cost = final_cost_summary.total_cost_usd - initial_cost.total_cost_usd
        final_total_calls = final_cost_summary.total_calls - initial_cost.total_calls
        
        # Use actual values from final summary if difference calculation fails
        if final_total_cost == 0 and final_cost_summary.total_cost_usd > 0:
            final_total_cost = final_cost_summary.total_cost_usd
        if final_total_calls == 0 and final_cost_summary.total_calls > 0:
            final_total_calls = final_cost_summary.total_calls
            
        print(f"   💰 GOLD generation cost: ${final_total_cost:.4f} ({final_total_calls} calls)")
        
        total_ranked_items = sum(len(ranking) for ranking in chunk_results.values())
        logger.info(f"GOLD: Generated ranking - {total_ranked_items} items")
        
        # Save GOLD ranking for analysis with sorted chunks
        sorted_chunk_results = {}
        for chunk_id in sorted(chunk_results.keys()):
            sorted_chunk_results[chunk_id] = chunk_results[chunk_id]
        
        gold_ranking_file = self.output_dir / "gold_ranking.json"
        with open(gold_ranking_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": model,
                "total_items": len(all_atomic_info),
                "total_chunks": len(chunk_atomic_groups),
                "ranking_items": total_ranked_items,
                "cost": final_total_cost,
                "calls": final_total_calls,
                "gold_ranking_by_chunk": sorted_chunk_results  # Sorted chunk-based structure
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"GOLD: Ranking saved")
        return chunk_results  # Return chunk-based structure (correctly ordered)
    
    def _generate_gold_ranking_chunk(
        self,
        chunk_atomic_list: List[AtomicInfo],
        doc_content: str,
        doc_title: str,
        language: str,
        model: str
    ) -> List[str]:
        """Generate GOLD ranking for a single chunk using improved idx-based prompt format"""
        
        if not chunk_atomic_list:
            return []
        
        try:
            # Build numbered atomic info list with title and content
            atomic_info_numbered_list = []
            for i, atomic_info in enumerate(chunk_atomic_list, 1):
                # Include document title if available
                title_prefix = getattr(atomic_info, 'document_title', '') or doc_title
                if title_prefix:
                    content_with_title = f"{title_prefix}: {atomic_info.content}"
                else:
                    content_with_title = atomic_info.content
                atomic_info_numbered_list.append(f"{i}: {content_with_title}")
            atomic_info_list_str = "\n".join(atomic_info_numbered_list)
            
            # Generate GOLD ranking prompt (with new idx-based format)
            gold_prompt = generate_prompt(
                PromptType.GENERATE_GOLD_RANKING,
                atomic_info_list=atomic_info_list_str,
                doc_title=doc_title,
                doc_content=doc_content,
                language=language
            )
            
            # Call LLM for GOLD ranking (with automatic retry logic from llm_client)
            gold_response = self.llm_client.call_api(gold_prompt, model)
            gold_parsed = clean_and_parse_json(gold_response)
            
            # Parse response with new idx-based format
            if isinstance(gold_parsed, dict) and "ranking" in gold_parsed:
                gold_ranking_data = gold_parsed["ranking"]
                if isinstance(gold_ranking_data, list):
                    # Extract atomic_info content based on item_index (new format)
                    chunk_ranking = []
                    for rank_item in gold_ranking_data:
                        if isinstance(rank_item, dict) and "item_index" in rank_item:
                            item_idx = rank_item["item_index"]
                            # Convert from 1-based to 0-based indexing
                            if 1 <= item_idx <= len(chunk_atomic_list):
                                atomic_content = chunk_atomic_list[item_idx - 1].content
                                chunk_ranking.append(atomic_content)
                            else:
                                logger.warning(f"[RARE GOLD] Invalid item_index {item_idx} for chunk with {len(chunk_atomic_list)} items")
                    
                    logger.debug(f"[RARE GOLD] Successfully ranked {len(chunk_ranking)}/{len(chunk_atomic_list)} items")
                    return chunk_ranking
                else:
                    logger.warning(f"[RARE GOLD] Invalid ranking data format: expected list, got {type(gold_ranking_data)}")
                    return []
            else:
                logger.warning(f"[RARE GOLD] Invalid response format: no 'ranking' field found")
                return []
                
        except Exception as e:
            logger.error(f"[RARE GOLD] Error generating GOLD ranking for chunk: {e}")
            return []

    def compare_method_rankings(
        self,
        method_rankings: Dict[str, List[str]],
        gold_ranking: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compare all validation method rankings against GOLD standard"""
        
        logger.info("[RARE GOLD] Comparing method rankings against GOLD standard")
        
        comparison_results = {}
        
        for method_name, method_ranking in method_rankings.items():
            logger.info(f"[RARE GOLD] Analyzing {method_name} ranking quality...")
            
            # Calculate comprehensive ranking metrics
            results = comprehensive_ranking_comparison(
                method_ranking=method_ranking,
                gold_ranking=gold_ranking,
                method_name=method_name
            )
            
            comparison_results[method_name] = results
        
        # Save comparison results
        comparison_file = self.output_dir / "ranking_comparison_results.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump({
                "gold_ranking_length": len(gold_ranking),
                "method_results": comparison_results,
                "ranking_metrics_explanation": {
                    "kendall_tau": "Ordinal association (-1 to 1), higher is better",
                    "spearman_correlation": "Monotonic relationship (-1 to 1), higher is better", 
                    "top_3_accuracy": "Fraction of top-3 items matching GOLD (0 to 1), higher is better",
                    "top_5_accuracy": "Fraction of top-5 items matching GOLD (0 to 1), higher is better",
                    "ndcg_3": "Normalized DCG@3 (0 to 1), higher is better",
                    "ranking_distance": "Normalized distance (0 to 1), lower is better",
                    "overall_quality_score": "Weighted average quality (0 to 1), higher is better"
                }
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[RARE GOLD] Ranking comparison results saved to {comparison_file}")
        
        return comparison_results

    def compare_method_rankings_chunk_based(
        self,
        method_chunk_rankings: Dict[str, Dict[str, List[str]]],
        gold_chunk_rankings: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare all validation method rankings against GOLD standard using chunk-based approach"""
        
        logger.info("[RARE GOLD] Comparing method rankings against GOLD standard (chunk-based)")
        
        # Import chunk-based comparison function
        from rare_core.rare_ranking_utils import comprehensive_chunk_based_comparison
        
        comparison_results = {}
        
        for method_name, method_rankings in method_chunk_rankings.items():
            logger.info(f"[RARE GOLD] Analyzing {method_name} ranking quality (chunk-based)...")
            
            # Calculate comprehensive chunk-based ranking metrics
            results = comprehensive_chunk_based_comparison(
                method_chunk_rankings=method_rankings,
                gold_chunk_rankings=gold_chunk_rankings,
                method_name=method_name
            )
            
            comparison_results[method_name] = results
        
        # Save chunk-based comparison results
        comparison_file = self.output_dir / "chunk_based_ranking_comparison_results.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump({
                "comparison_type": "chunk_based",
                "total_chunks": len(gold_chunk_rankings),
                "method_results": comparison_results,
                "ranking_metrics_explanation": {
                    "kendall_tau": "Average ordinal association (-1 to 1) across chunks, higher is better",
                    "spearman_correlation": "Average monotonic relationship (-1 to 1) across chunks, higher is better", 
                    "top_3_accuracy": "Average fraction of top-3 items matching GOLD across chunks (0 to 1), higher is better",
                    "top_5_accuracy": "Average fraction of top-5 items matching GOLD across chunks (0 to 1), higher is better",
                    "ndcg_3": "Average Normalized DCG@3 across chunks (0 to 1), higher is better",
                    "ranking_distance": "Average normalized distance across chunks (0 to 1), lower is better",
                    "overall_quality_score": "Average weighted quality score across chunks (0 to 1), higher is better",
                    "chunks_compared": "Number of chunks with sufficient data for comparison"
                }
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[RARE GOLD] Chunk-based comparison results saved to {comparison_file}")
        
        return comparison_results
    
    def _contains_excluded_keywords(self, content: str) -> bool:
        """Check if content contains excluded keywords (hard filter)"""
        content_lower = content.lower()
        return any(keyword.lower() in content_lower for keyword in EXCLUDE_KEYWORDS)
    
    def _validate_bulk_atomic_info(
        self, 
        atomic_info_list: List[AtomicInfo], 
        doc_content: str, 
        doc_title: str,
        language: str, 
        model: str,
        fewshot_examples: Optional[str] = None
    ) -> List[AtomicInfo]:
        """Validate multiple atomic information units using bulk comprehensive scoring system"""
        
        if not atomic_info_list:
            return atomic_info_list
        
        try:
            # Process ALL atomic info (no top-k pre-filtering for proper ranking)
            logger.info(f"[Step 4] Processing all {len(atomic_info_list)} atomic info items in chunk for comprehensive validation")
            
            # Build numbered atomic info list for prompt with title and content
            atomic_info_numbered_list = []
            for i, atomic_info in enumerate(atomic_info_list, 1):
                # Include document title if available
                title_prefix = getattr(atomic_info, 'document_title', '') or doc_title
                if title_prefix:
                    content_with_title = f"{title_prefix}: {atomic_info.content}"
                else:
                    content_with_title = atomic_info.content
                atomic_info_numbered_list.append(f"{i}. \"{content_with_title}\"")
            atomic_info_list_str = "\n".join(atomic_info_numbered_list)
            
            # Generate validation prompt (Combined) with optional few-shot examples
            prompt = generate_prompt(
                PromptType.VERIFY_ATOMIC_INFO_VALIDITY,
                atomic_info_list=atomic_info_list_str,
                atomic_count=len(atomic_info_list),
                doc_title=doc_title,
                doc_content=doc_content,
                few_shot_examples=fewshot_examples or "",
                language=language
            )
            
            # Call LLM for bulk validation with score completeness validation
            response = self.llm_client.call_api_with_score_validation(
                prompt=prompt, 
                validation_type='combined', 
                expected_count=len(atomic_info_list),
                model=model,
                max_retries=3
            )
            
            # DEBUG: Log raw response
            logger.info(f"[DEBUG Combined] LLM response length: {len(response) if response else 0} chars")
            
            # Parse response (expecting JSON array)
            parsed_response = clean_and_parse_json(response)
            
            # DEBUG: Log parsed response info
            logger.info(f"[DEBUG Combined] Parsed response type: {type(parsed_response)}, length: {len(parsed_response) if isinstance(parsed_response, list) else 'N/A'}")
            
            if not isinstance(parsed_response, list):
                logger.error(f"[DEBUG Combined] Expected JSON array, got {type(parsed_response)}")
                raise ValueError(f"Expected JSON array, got {type(parsed_response)}")
            
            if len(parsed_response) != len(atomic_info_list):
                logger.warning(f"[Step 3.5] Response count mismatch: expected {len(atomic_info_list)}, got {len(parsed_response)}")
                logger.debug(f"[DEBUG Combined] This chunk will have missing scores for some items")
            
            # Update atomic_info objects with validation results (only top items)
            for i, atomic_info in enumerate(atomic_info_list):
                if i < len(parsed_response):
                    result = parsed_response[i]
                    
                    # Extract individual scores
                    validity_score = result.get("validity_score", 0.0)
                    completeness_score = result.get("completeness_score", 0.0)
                    specificity_score = result.get("specificity_score", 0.0)
                    clarity_score = result.get("clarity_score", 0.0)
                    questionability_score = result.get("questionability_score", 0.0)
                    reasoning = result.get("reasoning", "No reasoning provided")
                    
                    # DEBUG: Log individual score extraction
                    available_keys = list(result.keys()) if isinstance(result, dict) else []
                    logger.debug(f"[DEBUG Combined] Item {i+1} available keys: {available_keys}")
                    logger.debug(f"[DEBUG Combined] Item {i+1} scores: v={validity_score}, c={completeness_score}, s={specificity_score}, cl={clarity_score}, q={questionability_score}")
                    
                    # Calculate overall_confidence as average of all 5 scores
                    scores = [validity_score, completeness_score, specificity_score, clarity_score, questionability_score]
                    overall_confidence = sum(scores) / len(scores) if scores else 0.0
                    
                    # Update atomic_info with validation results
                    atomic_info.validity_score = validity_score
                    atomic_info.completeness_score = completeness_score
                    atomic_info.specificity_score = specificity_score
                    atomic_info.clarity_score = clarity_score
                    atomic_info.questionability_score = questionability_score
                    atomic_info.overall_confidence = overall_confidence
                    atomic_info.validation_reasoning = reasoning
                else:
                    # No response for this top item, set zero scores
                    atomic_info.validity_score = 0.0
                    atomic_info.completeness_score = 0.0
                    atomic_info.specificity_score = 0.0
                    atomic_info.clarity_score = 0.0
                    atomic_info.questionability_score = 0.0
                    atomic_info.overall_confidence = 0.0
                    atomic_info.validation_reasoning = "Missing validation response for top item"
            
            return atomic_info_list
            
        except Exception as e:
            logger.warning(f"[Step 3.5] Bulk validation error: {e}")
            # Return original atomic_info_list with zero scores on error
            for atomic_info in atomic_info_list:
                atomic_info.validity_score = 0.0
                atomic_info.completeness_score = 0.0
                atomic_info.specificity_score = 0.0
                atomic_info.clarity_score = 0.0
                atomic_info.questionability_score = 0.0
                atomic_info.overall_confidence = 0.0
                atomic_info.validation_reasoning = f"Bulk validation error: {str(e)}"
            return atomic_info_list

    def _validate_separate_chunk_atomic_info(
        self, 
        atomic_info_list: List[AtomicInfo], 
        doc_content: str, 
        doc_title: str,
        language: str, 
        model: str,
        chunk_id: str = "",
        fewshot_examples: Optional[str] = None
    ) -> List[AtomicInfo]:
        """Validate atomic info list from one chunk using 5 separate LLM calls (chunk-based RARE-style)"""
        
        try:
            # Prepare atomic info list for prompts
            atomic_info_numbered_list = []
            for i, atomic_info in enumerate(atomic_info_list, 1):
                # Include document title if available
                title_prefix = getattr(atomic_info, 'document_title', '') or doc_title
                if title_prefix:
                    content_with_title = f"{title_prefix}: {atomic_info.content}"
                else:
                    content_with_title = atomic_info.content
                atomic_info_numbered_list.append(f"{i}. \"{content_with_title}\"")
            atomic_info_list_str = "\n".join(atomic_info_numbered_list)
            
            validity_prompt = generate_prompt(
                PromptType.VERIFY_ATOMIC_INFO_VALIDITY_SEPARATE,
                atomic_info_list=atomic_info_list_str,
                atomic_count=len(atomic_info_list),
                doc_title=doc_title,
                doc_content=doc_content,
                few_shot_examples=fewshot_examples or "",
                language=language
            )
            validity_response = self.llm_client.call_api_with_score_validation(
                prompt=validity_prompt,
                validation_type='separate_validity',
                expected_count=len(atomic_info_list),
                model=model,
                max_retries=3
            )
            validity_parsed = clean_and_parse_json(validity_response)
            validity_results = validity_parsed if isinstance(validity_parsed, list) else []
        
            # Call 2: Completeness validation for all items in chunk
            completeness_prompt = generate_prompt(
                PromptType.VERIFY_ATOMIC_INFO_COMPLETENESS_SEPARATE,
                atomic_info_list=atomic_info_list_str,
                atomic_count=len(atomic_info_list),
                doc_title=doc_title,
                doc_content=doc_content,
                few_shot_examples=fewshot_examples or "",
                language=language
            )
            completeness_response = self.llm_client.call_api_with_score_validation(
                prompt=completeness_prompt,
                validation_type='separate_completeness',
                expected_count=len(atomic_info_list),
                model=model,
                max_retries=3
            )
            completeness_parsed = clean_and_parse_json(completeness_response)
            completeness_results = completeness_parsed if isinstance(completeness_parsed, list) else []
        
            # Call 3: Specificity validation for all items in chunk
            specificity_prompt = generate_prompt(
                PromptType.VERIFY_ATOMIC_INFO_SPECIFICITY_SEPARATE,
                atomic_info_list=atomic_info_list_str,
                atomic_count=len(atomic_info_list),
                doc_title=doc_title,
                doc_content=doc_content,
                few_shot_examples=fewshot_examples or "",
                language=language
            )
            specificity_response = self.llm_client.call_api_with_score_validation(
                prompt=specificity_prompt,
                validation_type='separate_specificity',
                expected_count=len(atomic_info_list),
                model=model,
                max_retries=3
            )
            specificity_parsed = clean_and_parse_json(specificity_response)
            specificity_results = specificity_parsed if isinstance(specificity_parsed, list) else []
        
            # Call 4: Clarity validation for all items in chunk
            clarity_prompt = generate_prompt(
                PromptType.VERIFY_ATOMIC_INFO_CLARITY_SEPARATE,
                atomic_info_list=atomic_info_list_str,
                atomic_count=len(atomic_info_list),
                doc_title=doc_title,
                doc_content=doc_content,
                few_shot_examples=fewshot_examples or "",
                language=language
            )
            clarity_response = self.llm_client.call_api_with_score_validation(
                prompt=clarity_prompt,
                validation_type='separate_clarity',
                expected_count=len(atomic_info_list),
                model=model,
                max_retries=3
            )
            clarity_parsed = clean_and_parse_json(clarity_response)
            clarity_results = clarity_parsed if isinstance(clarity_parsed, list) else []
        
            # Call 5: Questionability validation for all items in chunk
            questionability_prompt = generate_prompt(
                PromptType.VERIFY_ATOMIC_INFO_QUESTIONABILITY_SEPARATE,
                atomic_info_list=atomic_info_list_str,
                atomic_count=len(atomic_info_list),
                doc_title=doc_title,
                doc_content=doc_content,
                few_shot_examples=fewshot_examples or "",
                language=language
            )
            questionability_response = self.llm_client.call_api_with_score_validation(
                prompt=questionability_prompt,
                validation_type='separate_questionability',
                expected_count=len(atomic_info_list),
                model=model,
                max_retries=3
            )
            questionability_parsed = clean_and_parse_json(questionability_response)
            questionability_results = questionability_parsed if isinstance(questionability_parsed, list) else []
            
            # Combine results from all 5 validations for each atomic info item
            validated_atomic_info_list = []
            for i, atomic_info in enumerate(atomic_info_list):
                try:
                    # Extract scores for this item (index i)
                    validity_score = 0.0
                    validity_reasoning = "No validity reasoning provided"
                    if i < len(validity_results) and isinstance(validity_results[i], dict):
                        validity_score = validity_results[i].get("validity_score", 0.0)
                        validity_reasoning = validity_results[i].get("reasoning", "No validity reasoning provided")
                    
                    completeness_score = 0.0
                    completeness_reasoning = "No completeness reasoning provided"
                    if i < len(completeness_results) and isinstance(completeness_results[i], dict):
                        completeness_score = completeness_results[i].get("completeness_score", 0.0)
                        completeness_reasoning = completeness_results[i].get("reasoning", "No completeness reasoning provided")
                    
                    specificity_score = 0.0
                    specificity_reasoning = "No specificity reasoning provided"
                    if i < len(specificity_results) and isinstance(specificity_results[i], dict):
                        specificity_score = specificity_results[i].get("specificity_score", 0.0)
                        specificity_reasoning = specificity_results[i].get("reasoning", "No specificity reasoning provided")
                    
                    clarity_score = 0.0
                    clarity_reasoning = "No clarity reasoning provided"
                    if i < len(clarity_results) and isinstance(clarity_results[i], dict):
                        clarity_score = clarity_results[i].get("clarity_score", 0.0)
                        clarity_reasoning = clarity_results[i].get("reasoning", "No clarity reasoning provided")
                    
                    questionability_score = 0.0
                    questionability_reasoning = "No questionability reasoning provided"
                    if i < len(questionability_results) and isinstance(questionability_results[i], dict):
                        questionability_score = questionability_results[i].get("questionability_score", 0.0)
                        questionability_reasoning = questionability_results[i].get("reasoning", "No questionability reasoning provided")
                    
                    # Calculate overall_confidence as average of all 5 scores
                    scores = [validity_score, completeness_score, specificity_score, clarity_score, questionability_score]
                    overall_confidence = sum(scores) / len(scores) if scores else 0.0
                    
                    # Combine reasoning from all 5 validations
                    combined_reasoning = f"VALIDITY: {validity_reasoning}\n\nCOMPLETENESS: {completeness_reasoning}\n\nSPECIFICITY: {specificity_reasoning}\n\nCLARITY: {clarity_reasoning}\n\nQUESTIONABILITY: {questionability_reasoning}"
                    
                    # Update atomic_info with validation results
                    atomic_info.validity_score = validity_score
                    atomic_info.completeness_score = completeness_score
                    atomic_info.specificity_score = specificity_score
                    atomic_info.clarity_score = clarity_score
                    atomic_info.questionability_score = questionability_score
                    atomic_info.overall_confidence = overall_confidence
                    atomic_info.validation_reasoning = combined_reasoning
                    
                    validated_atomic_info_list.append(atomic_info)
                    
                except Exception as item_error:
                    logger.warning(f"[Step 3.5] Error processing item {i} in chunk {chunk_id}: {item_error}")
                    # Set zero scores on error
                    atomic_info.validity_score = 0.0
                    atomic_info.completeness_score = 0.0
                    atomic_info.specificity_score = 0.0
                    atomic_info.clarity_score = 0.0
                    atomic_info.questionability_score = 0.0
                    atomic_info.overall_confidence = 0.0
                    atomic_info.validation_reasoning = f"Item validation error: {str(item_error)}"
                    validated_atomic_info_list.append(atomic_info)
            
            return validated_atomic_info_list
            
        except Exception as e:
            logger.warning(f"[Step 3.5] Chunk separate validation error for {chunk_id}: {e}")
            # Return all atomic_info with zero scores on error
            for atomic_info in atomic_info_list:
                atomic_info.validity_score = 0.0
                atomic_info.completeness_score = 0.0
                atomic_info.specificity_score = 0.0
                atomic_info.clarity_score = 0.0
                atomic_info.questionability_score = 0.0
                atomic_info.overall_confidence = 0.0
                atomic_info.validation_reasoning = f"Chunk validation error: {str(e)}"
            return atomic_info_list



    def _precompute_similarities(self, search_client, all_atomic_info: List[AtomicInfo], similarity_threshold: float, similarity_batch_size: int = 32) -> Dict[str, Any]:
        """Precompute similarity data using batched matrix multiplication + multithreading"""
        
        total_items = len(all_atomic_info)
        embeddings_tensor = torch.from_numpy(search_client.doc_embeddings).float()
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
        
        # Create mapping from atomic_info_id to embedding index
        id_to_embedding_idx = {}
        for idx, doc in enumerate(search_client.documents):
            id_to_embedding_idx[doc["atomic_info_id"]] = idx
        
        total_batches = (total_items + similarity_batch_size - 1) // similarity_batch_size
        similarity_data = {}
        
        with tqdm(total=total_batches, desc="Computing Batch Similarities") as pbar:
            for batch_idx, batch_start in enumerate(range(0, total_items, similarity_batch_size)):
                batch_end = min(batch_start + similarity_batch_size, total_items)
                batch_atomic_infos = all_atomic_info[batch_start:batch_end]
                
                # Get embedding indices for this batch
                batch_embedding_indices = []
                valid_batch_items = []
                for atomic_info in batch_atomic_infos:
                    if atomic_info.atomic_info_id in id_to_embedding_idx:
                        batch_embedding_indices.append(id_to_embedding_idx[atomic_info.atomic_info_id])
                        valid_batch_items.append(atomic_info)
                
                if not valid_batch_items:
                    pbar.update(1)
                    continue
                
                # Batch matrix multiplication using correct embedding indices
                batch_embeddings = embeddings_tensor[batch_embedding_indices]
                batch_similarities = torch.mm(batch_embeddings, embeddings_tensor.T)
                
                # Process results for each item in the batch
                batch_results = {}
                for i, atomic_info in enumerate(valid_batch_items):
                    similarities = batch_similarities[i]
                    embedding_idx = batch_embedding_indices[i]
                    
                    similar_items = []
                    for idx, score in enumerate(similarities):
                        score_val = score.item()
                        if score_val >= similarity_threshold and idx != embedding_idx:
                            document = search_client.documents[idx]
                            similar_items.append({
                                "atomic_info_id": document["atomic_info_id"],
                                "content": document["content"],
                                "chunk_id": document["chunk_id"],
                                "score": score_val
                            })
                    
                    # Filter out same-chunk items
                    valid_items = []
                    for similar_item in similar_items:
                        if similar_item["chunk_id"] != atomic_info.chunk_id:
                            valid_items.append({
                                "atomic_info_id": similar_item["atomic_info_id"],
                                "content": similar_item["content"],
                                "chunk_id": similar_item["chunk_id"],
                                "score": similar_item["score"]
                            })
                    
                    batch_results[atomic_info.atomic_info_id] = {
                        "target_content": atomic_info.content,
                        "target_chunk_id": atomic_info.chunk_id,
                        "valid_items": valid_items
                    }
                
                similarity_data.update(batch_results)
                pbar.update(1)
        
        items_with_similarities = sum(1 for data in similarity_data.values() if data["valid_items"])
        total_similarities = sum(len(data["valid_items"]) for data in similarity_data.values())
        
        logger.info(f"Step 5: Precomputed similarities - {items_with_similarities}/{total_items} have similarities")
        logger.info(f"Step 5: Total similarity pairs: {total_similarities}")
        
        return similarity_data
    
    def _process_batch_similarities(self, batch_similarities, batch_atomic_infos: List[AtomicInfo], all_atomic_info: List[AtomicInfo], threshold: float, batch_start: int, search_documents: List[Dict]) -> Dict[str, Any]:
        """Process batch similarities using multithreading"""
        
        def process_single_row(row_idx):
            similarities = batch_similarities[row_idx]
            atomic_info = batch_atomic_infos[row_idx]
            global_idx = batch_start + row_idx
            
            similar_items = []
            for idx, score in enumerate(similarities):
                score_val = score.item()
                if score_val >= threshold and idx != global_idx:
                    # Use search_documents directly - no index mismatch!
                    document = search_documents[idx]
                    similar_items.append({
                        "atomic_info_id": document["atomic_info_id"],
                        "content": document["content"],
                        "chunk_id": document["chunk_id"],
                        "score": score_val
                    })
            
            valid_items = []
            for similar_item in similar_items:
                if similar_item["chunk_id"] != atomic_info.chunk_id:
                    valid_items.append({
                        "atomic_info_id": similar_item["atomic_info_id"],
                        "content": similar_item["content"],
                        "chunk_id": similar_item["chunk_id"],
                        "score": similar_item["score"]
                    })
            
            return atomic_info.atomic_info_id, {
                "target_content": atomic_info.content,
                "target_chunk_id": atomic_info.chunk_id,
                "valid_items": valid_items
            }
        
        batch_results = {}
        batch_size = len(batch_atomic_infos)
        
        with ThreadPoolExecutor(max_workers=min(4, batch_size)) as executor:
            futures = [executor.submit(process_single_row, i) for i in range(batch_size)]
            for future in as_completed(futures):
                atomic_info_id, result_data = future.result()
                batch_results[atomic_info_id] = result_data
        
        return batch_results
    
    def _process_atomic_info(self, i: int, atomic_info: AtomicInfo, embeddings_tensor, all_atomic_info: List[AtomicInfo], threshold: float) -> Dict[str, Any]:
        """Process a single atomic_info item for similarity computation"""
        # Calculate cosine similarity directly using existing embeddings
        query_embedding = embeddings_tensor[i:i+1]  # Current item's embedding
        similarities = torch.mm(query_embedding, embeddings_tensor.T).squeeze()
        
        # Get ALL items above threshold (no top_k clipping)
        similar_items = []
        for idx, score in enumerate(similarities):
            if score.item() >= threshold and idx != i:  # Skip self-matching
                similar_ai = all_atomic_info[idx]
                similar_items.append({
                    "atomic_info_id": similar_ai.atomic_info_id,
                    "content": similar_ai.content,
                    "chunk_id": similar_ai.chunk_id,
                    "score": score.item()
                })
        
        # Filter valid items for LLM verification (same logic as step6)
        valid_items = []
        
        for similar_item in similar_items:
            similar_id = similar_item.get("atomic_info_id", "")
            similarity_score = similar_item.get("score", 0.0)
            
            # Skip self-matching
            if similar_id == atomic_info.atomic_info_id:
                continue
            
            # Skip items from the same chunk
            similar_chunk_id = similar_item.get("chunk_id", "")
            if similar_chunk_id == atomic_info.chunk_id:
                continue
            
            # Add to valid items (exact format that goes to LLM)
            valid_items.append({
                "atomic_info_id": similar_id,
                "content": similar_item.get("content", ""),
                "chunk_id": similar_chunk_id,
                "score": similarity_score
            })
        
        # Return the precomputed data for this item
        return {
            "target_content": atomic_info.content,
            "target_chunk_id": atomic_info.chunk_id,
            "valid_items": valid_items
        }

    def _calculate_auto_threshold(self, search_client, batch_size: int = 1024) -> float:
        """Calculate automatic threshold using batch processing for memory efficiency"""
        
        embeddings_tensor = torch.from_numpy(search_client.doc_embeddings).float()
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
        n = embeddings_tensor.shape[0]
        
        logger.info(f"Computing auto threshold from {n} embeddings (batch size: {batch_size})")
        
        total_similarity = 0.0
        total_pairs = 0
        
        # Process in batches to manage memory - computes full similarity matrix
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            
            for j in range(i, n, batch_size):
                j_end = min(j + batch_size, n)
                
                # Calculate similarities for this batch
                batch_i = embeddings_tensor[i:i_end]
                batch_j = embeddings_tensor[j:j_end]
                similarities = torch.mm(batch_i, batch_j.t())
                
                if i == j:
                    # Same batch: only upper triangle (excluding diagonal)
                    mask = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
                    batch_similarities = similarities[mask]
                else:
                    # Different batches: all pairs
                    batch_similarities = similarities.flatten()
                
                # Accumulate results
                total_similarity += batch_similarities.sum().item()
                total_pairs += batch_similarities.numel()
        
        # Calculate mean similarity as threshold
        auto_threshold = total_similarity / total_pairs if total_pairs > 0 else 0.0
        logger.info(f"Auto threshold: {auto_threshold:.6f} from {total_pairs:,} pairs")
        print(f"Auto threshold calculated: {auto_threshold:.6f}")
        
        return auto_threshold

    def step5_build_embeddings(self, atomic_info_map: Dict[str, List[AtomicInfo]], similarity_threshold: Union[float, str] = "auto", auto_threshold_batch_size: int = 1024, embedding_batch_size: int = 2048, similarity_only: bool = False, embedding_only: bool = False, similarity_batch_size: int = 32) -> Dict[str, Any]:
        """Step 5: Build embeddings for atomic information"""
        
        logger.info(f"[RARE Step 5] Building Embeddings - Processing atomic info")
        
        # Flatten all atomic info
        all_atomic_info = []
        for chunk_id, atomic_list in atomic_info_map.items():
            all_atomic_info.extend(atomic_list)
        
        if not all_atomic_info:
            raise ValueError("No atomic information available for embedding")
        
        # Create search documents for similarity calculation
        search_documents = []
        for atomic_info in all_atomic_info:
            doc = {
                "atomic_info_id": atomic_info.atomic_info_id,
                "chunk_id": atomic_info.chunk_id,
                "content": atomic_info.content,
            }
            search_documents.append(doc)
        
        # Initialize search client with atomic info as documents
        if similarity_only:
            # Load existing embeddings from pkl file
            existing_pkl_file = self.output_dir / "step4_embeddings.pkl"
            if not existing_pkl_file.exists():
                raise FileNotFoundError(f"Similarity only mode - embedding file not found: {existing_pkl_file}")
            
            with open(existing_pkl_file, 'rb') as f:
                existing_data = pickle.load(f)
            
            search_client = SearchClient(documents=[], api_key=os.getenv("OPENAI_API_KEY"), batch_size=embedding_batch_size)
            search_client.doc_embeddings = existing_data["embeddings"]
            search_client.documents = existing_data["search_documents"]
        else:
            search_client = SearchClient(search_documents, api_key=os.getenv("OPENAI_API_KEY"), batch_size=embedding_batch_size)
        
        # Store search client for cost tracking
        self.search_client = search_client
        
        # If embedding_only mode, save embeddings and calculate overall similarity average
        if embedding_only:
            embeddings_file = self.output_dir / "step4_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump({
                    "embeddings": search_client.doc_embeddings, 
                    "search_documents": search_client.documents,
                    "all_atomic_info": all_atomic_info
                }, f)
            logger.info(f"Embedding-only mode: Saved embeddings to {embeddings_file}")
            
            # Calculate overall similarity average for reference
            logger.info("Computing overall similarity average for threshold reference")
            overall_avg = self._calculate_auto_threshold(search_client, auto_threshold_batch_size)
            print(f"Overall similarity average: {overall_avg:.6f}")
            print("This can be used as reference for --similarity-only threshold")
            
            return {
                "total_embeddings": len(search_client.doc_embeddings), 
                "embeddings_shape": search_client.doc_embeddings.shape if hasattr(search_client.doc_embeddings, 'shape') else None,
                "overall_similarity_average": overall_avg
            }
        
        # Calculate threshold if auto is specified
        if similarity_threshold == "auto":
            logger.info(f"[RARE Step 5] Computing automatic similarity threshold")
            similarity_threshold = self._calculate_auto_threshold(search_client, auto_threshold_batch_size)
        
        # Precompute similarity data for step6
        logger.info(f"[RARE Step 5] Precomputing similarities for step6 (threshold: {similarity_threshold})")
        
        similarity_data = self._precompute_similarities(search_client, all_atomic_info, similarity_threshold, similarity_batch_size)
        
        # Save similarity data as PKL (more efficient for large data)
        similarity_file = self.output_dir / "precomputed_similarities.pkl"
        with open(similarity_file, 'wb') as f:
            pickle.dump(similarity_data, f)
        logger.info(f"[RARE Step 5] Similarity data saved to {similarity_file}")
        
        # Save embeddings and related data (avoid pickling the search_client directly)
        # Only save in non-similarity-only mode to avoid overwriting with sampled data
        output_file_pkl = self.output_dir / "step4_embeddings.pkl"
        if not similarity_only:
            with open(output_file_pkl, 'wb') as f:
                pickle.dump({
                    "embeddings": search_client.doc_embeddings,
                    "search_documents": search_documents,
                    "all_atomic_info": all_atomic_info,
                }, f)
        
        # Save metadata
        output_file_json = self.output_dir / "step4_embedding_data.json"
        with open(output_file_json, 'w', encoding='utf-8') as f:
            json.dump({
                "embeddings_shape": list(search_client.doc_embeddings.shape),
                "similarity_threshold": self.similarity_threshold,
                "total_atomic_info": len(all_atomic_info)
            }, f, ensure_ascii=False, indent=2)
        
        # Log cost information
        cost_info = search_client.get_usage_stats()
        logger.info(f"[RARE Step 5] Completed - Built embeddings for {len(all_atomic_info)} atomic units")
        logger.info(f"[RARE Step 5] Cost: ${cost_info['total_cost_usd']:.6f}, Tokens: {cost_info['total_tokens_used']}, Model: {cost_info['model']}")
        if similarity_only:
            logger.info(f"[Step 5] Saved: {output_file_json} (embeddings pkl preserved)")
        else:
            logger.info(f"[Step 5] Saved: {output_file_json}, {output_file_pkl}")
        
        return {
            "embeddings_file": str(output_file_pkl),
            "cost_info": cost_info,
            "total_embeddings": len(all_atomic_info),
            "embeddings_shape": list(search_client.doc_embeddings.shape)
        }
    
    def step6_redundancy_detection(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        embeddings_data: Union[Dict[str, Any], str],
        language: str,
        model: str,
        max_workers: int = 128,
        max_similar_items: int = None,
        top_k_per_chunk: int = None,
        max_chunks: int = None,
    ) -> Dict[str, RedundancyMapping]:
        """Step 6: Detect redundancies using precomputed similarities from Step 5"""
        
        logger.info(f"Step 6: Redundancy detection (using Step 5 results)")
        
        # Load embedding data
        if isinstance(embeddings_data, dict) and "embeddings_file" in embeddings_data:
            embeddings_file = embeddings_data["embeddings_file"]
        elif isinstance(embeddings_data, dict):
            # embeddings_data is a dict but not the expected format, find the pkl file
            pkl_files = list(self.output_dir.glob("step4_embeddings_*.pkl"))
            if not pkl_files:
                raise FileNotFoundError("No step4 embeddings file found")
            embeddings_file = str(max(pkl_files, key=lambda x: x.stat().st_mtime))
        else:
            embeddings_file = embeddings_data
            
        with open(embeddings_file, 'rb') as f:
            loaded_data = pickle.load(f)
            embeddings = loaded_data["embeddings"]
            # Convert numpy embeddings for compatibility
            if isinstance(embeddings, np.ndarray):
                pass  # Keep as numpy, conversion happens in search_client
            search_documents = loaded_data["search_documents"]
            all_atomic_info = loaded_data["all_atomic_info"]
        
        # Load precomputed similarity data from step5 output directory
        embeddings_file_path = Path(embeddings_file)
        similarity_file = embeddings_file_path.parent / "precomputed_similarities.pkl"
        with open(similarity_file, 'rb') as f:
            similarity_data = pickle.load(f)
        logger.info(f"Step 6: Loaded precomputed similarities from {similarity_file}")
        
        # Use only items that were processed in Step 5 (precomputed similarities)
        target_atomic_ids = set(similarity_data.keys())
        candidate_atomic_info = [ai for ai in all_atomic_info if ai.atomic_info_id in target_atomic_ids]

        # Optionally limit by top_k_per_chunk using Step 4 ranking file if available
        if top_k_per_chunk is not None and top_k_per_chunk > 0:
            # Try to locate Step 4 selection output next to embeddings directory
            step_root_dir = embeddings_file_path.parent.parent
            step4_dir = step_root_dir / "step4_output"
            step4_file_candidates = [
                step4_dir / "step4_selected_atomic_info.json",
                step4_dir / "selected_atomic_info.json",
            ]

            step4_file_path = None
            for cand in step4_file_candidates:
                if cand.exists():
                    step4_file_path = cand
                    break

            if step4_file_path is not None:
                try:
                    with open(step4_file_path, 'r', encoding='utf-8') as f:
                        step4_data = json.load(f)

                    # Prefer threshold_filtered mode
                    modes = step4_data.get("comparison_modes", {})
                    threshold_mode = modes.get("threshold_filtered", {})
                    ai_by_chunk = threshold_mode.get("atomic_info_by_chunk", {})
                    rankings_by_chunk = threshold_mode.get("chunk_rankings", {})

                    # Build content->id mapping per chunk
                    content_to_ids_by_chunk = {}
                    for cid, items in ai_by_chunk.items():
                        content_to_ids = {}
                        for item in items:
                            content = item.get("content")
                            aid = item.get("atomic_info_id")
                            if content is None or aid is None:
                                continue
                            content_to_ids.setdefault(content, []).append(aid)
                        content_to_ids_by_chunk[cid] = content_to_ids

                    # Collect allowed ids using ranking order (top-k per chunk)
                    allowed_ids = set()
                    for cid, ranked_contents in rankings_by_chunk.items():
                        top_contents = ranked_contents[:top_k_per_chunk]
                        content_to_ids = content_to_ids_by_chunk.get(cid, {})
                        for content in top_contents:
                            for aid in content_to_ids.get(content, []):
                                allowed_ids.add(aid)

                    # Fallback if ranking missing: take first k from atomic_info_by_chunk order
                    if not allowed_ids and ai_by_chunk:
                        for cid, items in ai_by_chunk.items():
                            for item in items[:top_k_per_chunk]:
                                aid = item.get("atomic_info_id")
                                if aid:
                                    allowed_ids.add(aid)

                    if allowed_ids:
                        candidate_atomic_info = [ai for ai in candidate_atomic_info if ai.atomic_info_id in allowed_ids]
                    else:
                        # If still nothing, fall back to score-based selection within chunk
                        chunk_to_items = {}
                        for ai in candidate_atomic_info:
                            cid = getattr(ai, "chunk_id", None)
                            if cid is None:
                                continue
                            chunk_to_items.setdefault(cid, []).append(ai)

                        limited_info = []
                        for cid, items in chunk_to_items.items():
                            def score_fn(x):
                                val = getattr(x, "overall_confidence", None)
                                if val is None:
                                    val = getattr(x, "validity_score", 0.0)
                                return float(val) if val is not None else 0.0

                            sorted_items = sorted(items, key=score_fn, reverse=True)
                            limited_info.extend(sorted_items[:top_k_per_chunk])
                        candidate_atomic_info = limited_info
                except Exception as e:
                    logger.warning(f"Step 6: Failed to read Step 4 selection file ({step4_file_path}): {e}. Falling back to score-based top-k.")
                    # Fallback to per-chunk score-based selection
                    chunk_to_items = {}
                    for ai in candidate_atomic_info:
                        cid = getattr(ai, "chunk_id", None)
                        if cid is None:
                            continue
                        chunk_to_items.setdefault(cid, []).append(ai)

                    limited_info = []
                    for cid, items in chunk_to_items.items():
                        def score_fn(x):
                            val = getattr(x, "overall_confidence", None)
                            if val is None:
                                val = getattr(x, "validity_score", 0.0)
                            return float(val) if val is not None else 0.0

                        sorted_items = sorted(items, key=score_fn, reverse=True)
                        limited_info.extend(sorted_items[:top_k_per_chunk])
                    candidate_atomic_info = limited_info
            else:
                # Step4 file not found; fallback to score-based selection
                chunk_to_items = {}
                for ai in candidate_atomic_info:
                    cid = getattr(ai, "chunk_id", None)
                    if cid is None:
                        continue
                    chunk_to_items.setdefault(cid, []).append(ai)

                limited_info = []
                for cid, items in chunk_to_items.items():
                    def score_fn(x):
                        val = getattr(x, "overall_confidence", None)
                        if val is None:
                            val = getattr(x, "validity_score", 0.0)
                        return float(val) if val is not None else 0.0

                    sorted_items = sorted(items, key=score_fn, reverse=True)
                    limited_info.extend(sorted_items[:top_k_per_chunk])
                candidate_atomic_info = limited_info
        
        # Apply max_similar_items filter if specified
        if max_similar_items is not None:
            filtered_atomic_info = []
            skipped_count = 0
            
            for ai in candidate_atomic_info:
                sim_data = similarity_data.get(ai.atomic_info_id, {})
                valid_items_count = len(sim_data.get("valid_items", []))
                
                if valid_items_count <= max_similar_items:
                    filtered_atomic_info.append(ai)
                else:
                    skipped_count += 1
            
            target_atomic_info = filtered_atomic_info
            logger.info(f"Step 6: Filtered by max_similar_items({max_similar_items}): {len(target_atomic_info)}/{len(candidate_atomic_info)} kept, {skipped_count} skipped")
        else:
            target_atomic_info = candidate_atomic_info
            logger.info(f"Step 6: Processing all {len(target_atomic_info)} items from Step 5 results")
        
        logger.info(f"Step 6: Found {len(target_atomic_ids)} precomputed similarities")
        
        # Process each target atomic info for redundancy with parallel processing
        redundancy_results = {}
        results_lock = threading.Lock()
        
        def process_atomic_info_worker(atomic_info):
            """Worker function to process a single atomic info item"""
            try:
                logger.debug(f"[Step 6] Processing redundancy: {atomic_info.atomic_info_id}")
                
                # Use precomputed similarity data (no more real-time similarity calculation)
                precomputed_data = similarity_data.get(atomic_info.atomic_info_id, {})
                valid_items = precomputed_data.get("valid_items", [])
                
                # Create similarity_scores for compatibility
                similarity_scores = {item["atomic_info_id"]: item["score"] for item in valid_items}
                
                # Prepare content with title for LLM verification (extract from chunk_id)
                chunk_id = atomic_info.chunk_id
                title = "_".join(chunk_id.split("_")[:-2]) if "_" in chunk_id else chunk_id
                if title:
                    content_with_title = f"{title}: {atomic_info.content}"
                else:
                    content_with_title = atomic_info.content
                
                # Batch LLM verification for semantic redundancy (1 API call instead of N calls)
                redundant_ids = self._verify_semantic_redundancy(
                    content_with_title,
                    valid_items,
                    language,
                    model
                )
                
                # Create mapping
                redundant_list = redundant_ids if redundant_ids else ["unique"]
                
                redundancy_mapping = RedundancyMapping(
                    atomic_info_id=atomic_info.atomic_info_id,
                    content=atomic_info.content,  # Store original content without title prefix
                    chunk_id=atomic_info.chunk_id,
                    redundant_items=redundant_list,
                    similarity_scores=similarity_scores
                )
                
                return redundancy_mapping
                
            except Exception as e:
                logger.error(f"[Step 6] Error processing {atomic_info.atomic_info_id}: {e}")
                return None
        
        # Process with ThreadPoolExecutor (using target atomic info only)
        with tqdm(total=len(target_atomic_info), desc="Detecting redundancies", unit="atomic_info") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks (targets only, but compare against full pool)
                future_to_atomic_info = {
                    executor.submit(process_atomic_info_worker, atomic_info): atomic_info 
                    for atomic_info in target_atomic_info
                }
                
                # Process completed tasks
                for future in as_completed(future_to_atomic_info):
                    atomic_info = future_to_atomic_info[future]
                    try:
                        redundancy_mapping = future.result()
                        if redundancy_mapping:
                            # Thread-safe updates (use consistent cost method like Step 4)
                            with results_lock:
                                redundancy_results[redundancy_mapping.atomic_info_id] = redundancy_mapping
                                # Calculate statistics inside lock for thread safety
                                current_cost = self.llm_client.get_cost_summary().total_cost_usd
                                unique_so_far = sum(1 for r in redundancy_results.values() if r.redundant_items == ["unique"])
                                pbar.set_postfix({
                                    'Cost': f'${current_cost:.4f}',
                                    'Unique': unique_so_far,
                                    'Redundant': len(redundancy_results) - unique_so_far
                                })
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"[Step 6] Error processing {atomic_info.atomic_info_id}: {e}")
                        pbar.update(1)
        
        # Save result (convert RedundancyMapping to dict for JSON)
        redundancy_dict = {}
        for atomic_id, mapping in redundancy_results.items():
            redundancy_dict[atomic_id] = {
                "atomic_info_id": mapping.atomic_info_id,
                "content": mapping.content,
                "chunk_id": mapping.chunk_id,
                "redundant_items": mapping.redundant_items,
                "similarity_scores": mapping.similarity_scores
            }
        
        output_file = self.output_dir / "step6_redundancy_mapping.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(redundancy_dict, f, ensure_ascii=False, indent=2)
        
        # Log statistics
        unique_count = sum(1 for r in redundancy_results.values() if r.redundant_items == ["unique"])
        redundant_count = len(redundancy_results) - unique_count
        logger.info(f"[RARE Step 6] Completed - Unique: {unique_count}, Redundant: {redundant_count}, saved to {output_file}")
        
        return redundancy_results
    
    def step7_multihop_question_generation(
        self,
        redundancy_mapping: Dict[str, RedundancyMapping],
        chunk_data: Dict[str, Dict[str, Any]],
        *,
        num_information: int = 2,
        num_sample: int = 10,
        num_questions: int = 10,
        input_pool_size: int = 100,
        generation_model: str = "gpt5",
        filter_model: str = "gpt5_nano",
        validation_model: str = "gpt5_nano",
        answerability_model: str = "gpt5_nano",
        language: str = "English",
        max_workers: int = 128,
        legacy_mode: bool = False,
        legacy_target_count: Optional[int] = None,
        legacy_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Step 7: Generate multi-hop questions using LLM selection pipeline."""

        if not redundancy_mapping:
            logger.warning("[RARE Step 7] Empty redundancy mapping provided")
            return {
                "generated_samples": [],
                "extended_samples": [],
                "failed_samples": {},
                "stats": {},
                "legacy_questions": [],
            }

        if legacy_mode:
            target = legacy_target_count if legacy_target_count is not None else num_sample
            legacy_model_name = legacy_model if legacy_model else generation_model
            questions = self._step7_legacy_hotpot_flow(
                redundancy_mapping=redundancy_mapping,
                target_count=target,
                model=legacy_model_name,
                language=language,
            )
            return {
                "generated_samples": [],
                "extended_samples": [],
                "failed_samples": {},
                "stats": {},
                "legacy_questions": questions,
            }

        diverse_items = self._step7_prepare_diverse_pool(
            redundancy_mapping=redundancy_mapping,
            num_information=num_information,
        )

        actual_num_sample = min(num_sample, len(diverse_items) // max(1, num_information))
        if actual_num_sample <= 0:
            logger.warning("[RARE Step 7] Not enough diverse items to generate samples")
            return {
                "generated_samples": [],
                "extended_samples": [],
                "failed_samples": {},
                "stats": {},
                "legacy_questions": [],
            }

        tracker = _Step7ValidationTracker()
        generation_args = SimpleNamespace(
            input_pool_size=input_pool_size,
            num_information=num_information,
            output_questions=num_questions,
            max_workers=max_workers,
            generation_model=generation_model,
            filter_model=filter_model,
            validation_model=validation_model,
            answerability_model=answerability_model,
            language=language,
        )

        generated_samples, extended_samples = self._step7_generate_llm_selection_samples(
            diverse_items=diverse_items,
            redundancy_mapping=redundancy_mapping,
            chunk_data_dict=chunk_data,
            args=generation_args,
            actual_num_sample=actual_num_sample,
            tracker=tracker,
        )

        artifacts = self._step7_persist_results(
            generated_samples=generated_samples,
            extended_samples=extended_samples,
            chunk_data_dict=chunk_data,
            num_information=num_information,
            tracker=tracker,
            legacy_mode=False,
        )

        return {
            "generated_samples": generated_samples,
            "extended_samples": extended_samples,
            "failed_samples": tracker.failed_samples,
            "stats": tracker.stats,
            "legacy_questions": [],
            "artifacts": artifacts,
        }

    def _step7_legacy_hotpot_flow(
        self,
        redundancy_mapping: Dict[str, RedundancyMapping],
        target_count: int,
        model: str,
        language: str,
    ) -> List[Dict[str, Any]]:
        logger.info("[RARE Step 7] Legacy HotPot pipeline starts")
        gold_sentence_groups = self._convert_redundancy_to_gold_sentences(redundancy_mapping, target_count)
        if not gold_sentence_groups:
            logger.warning("[RARE Step 7] Legacy flow found no redundancy groups")
            return []

        multihop_questions: List[Dict[str, Any]] = []
        total_groups = len(gold_sentence_groups)
        with tqdm(total=total_groups, desc="Generating multi-hop questions", unit="groups") as pbar:
            for index, gold_sentences in enumerate(gold_sentence_groups, 1):
                try:
                    generated_result = self.generate_multihop_questions(
                        gold_sentences=gold_sentences,
                        model=model,
                        max_workers=16,
                        num_candidates=10,
                    )
                    if not generated_result or "candidates" not in generated_result:
                        pbar.update(1)
                        continue

                    candidates = generated_result["candidates"]
                    filtered = self._apply_logical_consistency_filtering(candidates, gold_sentences, model)
                    if not filtered:
                        pbar.update(1)
                        continue

                    validated = self._apply_quality_validation(filtered, gold_sentences, model)
                    if not validated:
                        pbar.update(1)
                        continue

                    ranked_candidates = self._rrf_rank_questions(validated)
                    best_question = ranked_candidates[0] if ranked_candidates else None
                    if best_question:
                        best_question.update(
                            {
                                "group_id": f"group_{index}",
                                "source_redundancy_group": [item.atomic_info_id for item in gold_sentences],
                                "redundancy_level": len(gold_sentences),
                                "generation_method": "hotpot_multihop_pipeline",
                            }
                        )
                        multihop_questions.append(best_question)

                    current_cost = self.llm_client.get_cost_summary().total_cost_usd
                    pbar.set_postfix({"Cost": f"${current_cost:.4f}", "Generated": len(multihop_questions)})
                    pbar.update(1)
                except Exception as exc:
                    logger.error("[RARE Step 7] Legacy flow error for group %d: %s", index, exc)
                    pbar.update(1)
                    continue

        multihop_questions.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        final_questions = multihop_questions[:target_count]

        output_file = self.output_dir / "step7_multihop_questions.json"
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(final_questions, fp, ensure_ascii=False, indent=2)

        summary_file = self.output_dir / "step7_multihop_questions_summary.json"
        with open(summary_file, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "total_redundancy_groups_processed": total_groups,
                    "successful_generations": len(multihop_questions),
                    "final_selected_questions": len(final_questions),
                    "success_rate": len(multihop_questions) / total_groups if total_groups else 0,
                    "multihop_questions": final_questions,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "[RARE Step 7] Legacy completed - Generated %d questions (success %.1f%%)",
            len(final_questions),
            (len(multihop_questions) / total_groups * 100) if total_groups else 0,
        )
        return final_questions

    def _step7_persist_results(
        self,
        generated_samples: List[_Step7QuestionSample],
        extended_samples: List[_Step7QuestionSample],
        chunk_data_dict: Dict[str, Dict[str, Any]],
        num_information: int,
        tracker: _Step7ValidationTracker,
        legacy_mode: bool,
    ) -> Dict[str, Path]:
        artifacts: Dict[str, Path] = {}
        if legacy_mode or not generated_samples:
            logger.info("[RARE Step 7] No new samples to persist")
            return artifacts

        output_file = self.output_dir / f"multihop_questions_dataset_num{num_information}.json"
        artifacts["dataset"] = output_file
        samples_data = [asdict(sample) for sample in generated_samples]

        chunk_mapping: Dict[str, Dict[str, Any]] = {}
        for sample in generated_samples:
            for chunk_group in sample.gold_chunks:
                for chunk_id in chunk_group:
                    if chunk_id in chunk_data_dict and chunk_id not in chunk_mapping:
                        chunk_mapping[chunk_id] = chunk_data_dict[chunk_id]

        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "metadata": {
                        "total_samples": len(generated_samples),
                        "generation_timestamp": datetime.now().isoformat(),
                        "format": "RARE_QuestionSample",
                    },
                    "samples": samples_data,
                    "chunk_mapping": chunk_mapping,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

        if extended_samples:
            extended_file = self.output_dir / "extended_validated_samples.json"
            artifacts["extended_dataset"] = extended_file
            extended_data = [asdict(sample) for sample in extended_samples]
            extended_mapping: Dict[str, Dict[str, Any]] = {}
            for sample in extended_samples:
                for chunk_group in sample.gold_chunks:
                    for chunk_id in chunk_group:
                        if chunk_id in chunk_data_dict and chunk_id not in extended_mapping:
                            extended_mapping[chunk_id] = chunk_data_dict[chunk_id]

            with open(extended_file, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "metadata": {
                            "total_samples": len(extended_samples),
                            "generation_timestamp": datetime.now().isoformat(),
                            "format": "RARE_Extended",
                        },
                        "samples": extended_data,
                        "chunk_mapping": extended_mapping,
                    },
                    fp,
                    ensure_ascii=False,
                    indent=2,
                )

        if tracker.failed_samples:
            failed_file = self.output_dir / f"questions_failed_num{num_information}.json"
            artifacts["failed_samples"] = failed_file
            with open(failed_file, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "metadata": {
                            "total_failed_questions": len(tracker.failed_samples),
                            "failure_type_counts": {
                                logical_type: sum(
                                    1
                                    for sample in tracker.failed_samples.values()
                                    if logical_type in sample.get("failure_types", [])
                                )
                                for logical_type in list(STEP7_LOGICAL_TYPE_MAPPING.keys()) + ["answerability"]
                            },
                            "generation_timestamp": datetime.now().isoformat(),
                        },
                        "failed_samples": tracker.failed_samples,
                    },
                    fp,
                    ensure_ascii=False,
                    indent=2,
                )

        logger.info(
            "[RARE Step 7] Generated %d/%d samples (success rate %.1f%%)",
            len(generated_samples),
            tracker.progress.get("generation", 0),
            (len(generated_samples) / tracker.progress.get("generation", 1) * 100)
            if tracker.progress.get("generation", 0)
            else 0,
        )
        return artifacts

    # =============================================================================
    # Step 7 Helper Methods (HotPot Questions Integration)
    # =============================================================================
    
    def _convert_redundancy_to_gold_sentences(
        self, 
        redundancy_mapping: Dict[str, RedundancyMapping], 
        target_count: int
    ) -> List[List[Dict]]:
        """Convert redundancy mapping to gold_sentences format for HotPot Questions."""
        
        # Get redundant groups (groups with multiple similar items)
        redundant_groups = []
        for mapping in redundancy_mapping.values():
            if mapping.redundant_items != ["unique"] and len(mapping.redundant_items) >= 2:
                redundant_groups.append(mapping)
        
        # Sort by redundancy level (more redundant = higher priority)
        # Note: After Step 4, order represents importance (RRF-sorted), so we don't need importance_score
        redundant_groups.sort(
            key=lambda x: len(x.redundant_items), 
            reverse=True
        )
        
        # Convert to gold_sentences format
        gold_sentence_groups = []
        for mapping in redundant_groups[:target_count * 2]:  # Process more than target for better selection
            # Create gold sentences from redundant group
            gold_sentences = []
            
            # Add main atomic info
            gold_sentences.append({
                'atomic_info_id': mapping.atomic_info_id,
                'content': mapping.content,
                'chunk_id': mapping.chunk_id,
                'role': 'primary'
            })
            
            # Add related redundant items (simulate from redundant_items list)
            for i, redundant_id in enumerate(mapping.redundant_items[:3]):  # Max 3 additional
                if redundant_id != "unique":
                    # Find related redundant content (simplified - in real scenario would fetch from mapping)
                    gold_sentences.append({
                        'atomic_info_id': f"{mapping.atomic_info_id}_related_{i+1}",
                        'content': f"Related information to: {mapping.content[:100]}...",  # Simplified
                        'chunk_id': mapping.chunk_id,
                        'role': 'supporting'
                    })
            
            if len(gold_sentences) >= 2:  # Need at least 2 sentences for multi-hop
                gold_sentence_groups.append(gold_sentences)
        
        logger.info(f"[RARE Step 7] Converted {len(redundant_groups)} redundancy groups to {len(gold_sentence_groups)} gold sentence groups")
        return gold_sentence_groups
    
    def _apply_logical_consistency_filtering(
        self, 
        candidates: List[Dict], 
        gold_sentences: List[Dict], 
        model: str
    ) -> List[Dict]:
        """Apply 4-dimensional logical consistency filtering."""
        
        filtered_candidates = []
        
        for candidate in candidates:
            if 'generated_question' not in candidate:
                continue
                
            question = candidate['generated_question']
            answer = candidate.get('generated_answer', '')
            
            try:
                # Format sentences for prompt
                sentences_text = '\n'.join([f"- {s['content']}" for s in gold_sentences])
                
                # Apply all 4 logical consistency filters
                filters_passed = 0
                filter_results = {}
                
                # 1. Context Independence
                context_prompt = generate_prompt(
                    PromptType.FILTER_CONTEXT_ASSUMPTION,
                    question=question,
                    sentences=sentences_text
                )
                context_result = self.llm_client.call_api(context_prompt, model, validate_json=True)
                context_data = clean_and_parse_json(context_result)
                if context_data and context_data.get('result') == 'PASS':
                    filters_passed += 1
                filter_results['context_independence'] = context_data.get('result', 'FAIL')
                
                # 2. Information Completeness  
                completeness_prompt = generate_prompt(
                    PromptType.FILTER_INFORMATION_COMPLETENESS,
                    question=question,
                    answer=answer,
                    sentences=sentences_text
                )
                completeness_result = self.llm_client.call_api(completeness_prompt, model, validate_json=True)
                completeness_data = clean_and_parse_json(completeness_result)
                if completeness_data and completeness_data.get('result') == 'PASS':
                    filters_passed += 1
                filter_results['information_completeness'] = completeness_data.get('result', 'FAIL')
                
                # 3. Question Ambiguity
                ambiguity_prompt = generate_prompt(
                    PromptType.FILTER_QUESTION_AMBIGUITY,
                    question=question,
                    sentences=sentences_text
                )
                ambiguity_result = self.llm_client.call_api(ambiguity_prompt, model, validate_json=True)
                ambiguity_data = clean_and_parse_json(ambiguity_result)
                if ambiguity_data and ambiguity_data.get('result') == 'PASS':
                    filters_passed += 1
                filter_results['question_ambiguity'] = ambiguity_data.get('result', 'FAIL')
                
                # 4. Conversion Logic
                conversion_prompt = generate_prompt(
                    PromptType.FILTER_CONVERSION_LOGIC,
                    question=question,
                    answer=answer,
                    sentences=sentences_text
                )
                conversion_result = self.llm_client.call_api(conversion_prompt, model, validate_json=True)
                conversion_data = clean_and_parse_json(conversion_result)
                if conversion_data and conversion_data.get('result') == 'PASS':
                    filters_passed += 1
                filter_results['conversion_logic'] = conversion_data.get('result', 'FAIL')
                
                # Only pass if all 4 filters pass
                if filters_passed == 4:
                    candidate['logical_filtering_results'] = filter_results
                    candidate['logical_filters_passed'] = filters_passed
                    filtered_candidates.append(candidate)
                    
            except Exception as e:
                logger.debug(f"[Step 7] Error in logical filtering: {e}")
                continue
        
        logger.debug(f"[RARE Step 7] Logical filtering: {len(filtered_candidates)}/{len(candidates)} candidates passed")
        return filtered_candidates
    
    def _apply_quality_validation(
        self, 
        candidates: List[Dict], 
        gold_sentences: List[Dict], 
        model: str
    ) -> List[Dict]:
        """Apply 4-dimensional quality validation."""
        
        validated_candidates = []
        
        for candidate in candidates:
            if 'generated_question' not in candidate:
                continue
                
            question = candidate['generated_question']
            answer = candidate.get('generated_answer', '')
            
            try:
                # Format sentences for prompt
                sentences_text = '\n'.join([f"- {s['content']}" for s in gold_sentences])
                
                quality_scores = {}
                
                # 1. Connectivity validation
                connectivity_prompt = generate_prompt(
                    PromptType.VALIDATE_MULTIHOP_QUESTIONS_CONNECTIVITY_SEPARATE,
                    question=question,
                    answer=answer,
                    sentences=sentences_text
                )
                connectivity_result = self.llm_client.call_api(connectivity_prompt, model, validate_json=True)
                connectivity_data = clean_and_parse_json(connectivity_result)
                quality_scores['connectivity_score'] = connectivity_data.get('score', 0) if connectivity_data else 0
                
                # 2. Fluency validation
                fluency_prompt = generate_prompt(
                    PromptType.VALIDATE_MULTIHOP_QUESTIONS_FLUENCY_SEPARATE,
                    question=question,
                    answer=answer,
                    sentences=sentences_text
                )
                fluency_result = self.llm_client.call_api(fluency_prompt, model, validate_json=True)
                fluency_data = clean_and_parse_json(fluency_result)
                quality_scores['fluency_score'] = fluency_data.get('score', 0) if fluency_data else 0
                
                # 3. Essentiality validation
                essentiality_prompt = generate_prompt(
                    PromptType.VALIDATE_MULTIHOP_QUESTIONS_ESSENTIALITY_SEPARATE,
                    question=question,
                    answer=answer,
                    sentences=sentences_text
                )
                essentiality_result = self.llm_client.call_api(essentiality_prompt, model, validate_json=True)
                essentiality_data = clean_and_parse_json(essentiality_result)
                quality_scores['essentiality_score'] = essentiality_data.get('score', 0) if essentiality_data else 0
                
                # 4. Validity validation
                validity_prompt = generate_prompt(
                    PromptType.VALIDATE_MULTIHOP_QUESTIONS_VALIDITY_SEPARATE,
                    question=question,
                    answer=answer,
                    sentences=sentences_text
                )
                validity_result = self.llm_client.call_api(validity_prompt, model, validate_json=True)
                validity_data = clean_and_parse_json(validity_result)
                quality_scores['validity_score'] = validity_data.get('score', 0) if validity_data else 0
                
                # Add quality scores to candidate
                candidate.update(quality_scores)
                validated_candidates.append(candidate)
                
            except Exception as e:
                logger.debug(f"[Step 7] Error in quality validation: {e}")
                continue
        
        logger.debug(f"[RARE Step 7] Quality validation: {len(validated_candidates)}/{len(candidates)} candidates validated")
        return validated_candidates
    
    def _rrf_rank_questions(self, questions: List[Dict]) -> List[Dict]:
        """Rank questions using RRF based on 4 quality dimensions."""
        if len(questions) <= 1:
            if len(questions) == 1:
                questions[0]['rrf_score'] = 4.0
            return questions
        
        dimensions = ['connectivity_score', 'fluency_score', 'essentiality_score', 'validity_score']
        
        # Create dimension-based rankings
        dimension_rankings = {}
        for dim in dimensions:
            sorted_questions = sorted(questions, key=lambda x: x.get(dim, 0), reverse=True)
            dimension_rankings[dim] = {id(q): pos + 1 for pos, q in enumerate(sorted_questions)}
        
        # Calculate RRF scores
        for question in questions:
            q_id = id(question)
            rrf_score = sum(1.0 / dimension_rankings[dim][q_id] for dim in dimensions)
            question['rrf_score'] = rrf_score
        
        # Sort by RRF score (higher is better)
        ranked_questions = sorted(questions, key=lambda x: x['rrf_score'], reverse=True)
        
        logger.debug(f"[RARE Step 7] RRF ranking completed for {len(ranked_questions)} questions")
        return ranked_questions
    
    # =============================================================================
    # Utility Functions
    # =============================================================================
    
    def load_step_result(self, filename_pattern: str) -> Any:
        """Load result from previous step"""
        # Find the most recent file matching the pattern
        files = list(self.output_dir.glob(f"*{filename_pattern}*"))
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {filename_pattern}")
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Loading step result from: {latest_file}")
        
        if latest_file.suffix == '.json':
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert back to objects if needed
            if "atomic_info_map" in filename_pattern:
                # Convert dict back to AtomicInfo objects
                atomic_info_map = {}
                for chunk_id, atomic_list in data.items():
                    atomic_info_objects = []
                    for ai_dict in atomic_list:
                        atomic_info = AtomicInfo(
                            content=ai_dict["content"],
                            chunk_id=ai_dict["chunk_id"],
                            atomic_info_id=ai_dict["atomic_info_id"]
                        )
                        atomic_info_objects.append(atomic_info)
                    atomic_info_map[chunk_id] = atomic_info_objects
                return atomic_info_map
            elif "redundancy_mapping" in filename_pattern:
                # Convert dict back to RedundancyMapping objects
                redundancy_mapping = {}
                for atomic_id, mapping_dict in data.items():
                    mapping = RedundancyMapping(
                        atomic_info_id=mapping_dict["atomic_info_id"],
                        content=mapping_dict["content"],
                        chunk_id=mapping_dict["chunk_id"],
                        redundant_items=mapping_dict["redundant_items"],
                        similarity_scores=mapping_dict["similarity_scores"]
                    )
                    redundancy_mapping[atomic_id] = mapping
                return redundancy_mapping
            else:
                return data
        else:  # .pkl files
            with open(latest_file, 'rb') as f:
                return pickle.load(f)
    
    # =============================================================================
    # Internal Helper Functions (from original implementation)
    # =============================================================================
    
    def _generate_doc_id(self, doc: dict, index: int) -> str:
        """Generate document ID from metadata"""
        file_name = doc.get("file_name", f"doc_{index}").replace(".pdf", "")
        page_no = doc.get("page_no", 1)
        sub_text_index_raw = doc.get("sub_text_index", index)
        
        # Handle sub_text_index - could be string like "1-1" or integer
        if isinstance(sub_text_index_raw, str):
            try:
                sub_text_index = int(sub_text_index_raw.split('-')[0])
            except (ValueError, AttributeError):
                sub_text_index = index
        else:
            sub_text_index = int(sub_text_index_raw)
        
        return f"{file_name}_page{page_no:03d}_chunk{sub_text_index:03d}"
    
    def _extract_atomic_info_single(
        self, 
        content: str, 
        chunk_id: str, 
        language: str, 
        model: str,
        doc_title: str = ""
    ) -> List[AtomicInfo]:
        """Extract atomic information from a single document"""
        
        try:
            # Generate prompt and call LLM
            prompt = generate_prompt(
                PromptType.EXTRACT_ATOMIC_INFO,
                doc_title=doc_title,
                doc_content=content,
                language=language
            )
            
            response = self.llm_client.call_api(prompt, model=model)
            parsed_response = clean_and_parse_json(response)
            
            # Parse response
            if isinstance(parsed_response, dict) and "atomic_information" in parsed_response:
                atomic_info_list = parsed_response["atomic_information"]
                
                if isinstance(atomic_info_list, list):
                    valid_info = []
                    
                    for i, info_item in enumerate(atomic_info_list):
                        if isinstance(info_item, dict) and "content" in info_item:
                            content_text = info_item["content"].strip()
                            
                            if content_text and len(content_text) >= 10:  # Basic validation
                                atomic_info_id = f"{chunk_id}_atomic_{i:03d}"
                                atomic_info = AtomicInfo(
                                    content=content_text,
                                    chunk_id=chunk_id,
                                    atomic_info_id=atomic_info_id
                                )
                                valid_info.append(atomic_info)
                    
                    return valid_info
            
            logger.warning(f"[RARE] Invalid response format for atomic info extraction")
            return []
            
        except Exception as e:
            logger.error(f"[RARE] Error extracting atomic information: {e}")
            return []
    
    def _verify_semantic_redundancy(
        self,
        target_info: str,
        comparison_items: List[Dict[str, Any]],
        language: str,
        model: str
    ) -> List[str]:
        """Verify semantic redundancy between target and multiple comparison items (batch processing)"""
        
        if not comparison_items:
            return []
            
        try:
            # Format comparison info list for batch processing
            comparison_info_list = []
            for i, item in enumerate(comparison_items, 1):
                comparison_info_list.append(f"{i}: {item.get('content', '')}")
            
            comparison_info_formatted = "\n".join(comparison_info_list)
            
            prompt = generate_prompt(
                PromptType.DETECT_SEMANTIC_REDUNDANCY,
                target_info=target_info,
                comparison_info_list=comparison_info_formatted,
                num_comparisons=len(comparison_items)
            )
            
            response = self.llm_client.call_api(prompt, model=model)
            parsed_response = clean_and_parse_json(response)
            
            redundant_ids = []
            
            # Process batch response
            if isinstance(parsed_response, list):
                for i, result in enumerate(parsed_response):
                    if isinstance(result, dict) and result.get("is_redundant", False):
                        if i < len(comparison_items):
                            redundant_ids.append(comparison_items[i].get("atomic_info_id", ""))
            
            return redundant_ids
            
        except Exception as e:
            logger.error(f"[RARE] Error verifying semantic redundancy: {e}")
            return []
    
    def _generate_evaluation_item(
        self,
        redundancy_result: RedundancyMapping,
        model: str,
        language: str
    ) -> RareEvaluationItem:
        """Generate a single evaluation item from redundancy result"""
        
        # Generate question
        question_prompt = generate_prompt(
            PromptType.GENERATE_QUESTION_FROM_ATOMIC,
            atomic_info=redundancy_result.content,
            doc_source=redundancy_result.chunk_id,
            redundancy_level=len(redundancy_result.redundant_items) if redundancy_result.redundant_items != ["unique"] else 0,
            language=language
        )
        
        question_response = self.llm_client.call_api(question_prompt, model=model)
        question_parsed = clean_and_parse_json(question_response)
        question = question_parsed.get("question", f"What is the specific information about {redundancy_result.content[:50]}...?")
        
        # Generate answer
        answer_prompt = generate_prompt(
            PromptType.GENERATE_ANSWER_FROM_ATOMIC,
            question=question,
            atomic_info=redundancy_result.content,
            language=language
        )
        
        answer_response = self.llm_client.call_api(answer_prompt, model=model)
        answer_parsed = clean_and_parse_json(answer_response)
        answer = answer_parsed.get("answer", redundancy_result.content)
        
        # Create evaluation item
        return RareEvaluationItem(
            question=question,
            target_answer=answer,
            atomic_info=redundancy_result.content,
            atomic_info_id=redundancy_result.atomic_info_id,
            redundancy_level=len(redundancy_result.redundant_items) if redundancy_result.redundant_items != ["unique"] else 0,
            chunk_id=redundancy_result.chunk_id,
            document_source=redundancy_result.chunk_id,
            similar_atomic_info_ids=redundancy_result.redundant_items if redundancy_result.redundant_items != ["unique"] else []
        )
    
    def should_terminate(self, loop_count: int) -> bool:
        """Check if pipeline should terminate due to limits"""
        current_cost = self.llm_client.get_cost_summary().total_cost_usd
        current_calls = self.llm_client.get_total_api_calls()

        if current_cost >= MAX_COST_USD:
            logger.info(f"[RARE] Terminating due to cost limit: ${current_cost:.4f} >= ${MAX_COST_USD}")
            return True
        elif loop_count >= MAX_LOOP_COUNT:
            logger.info(f"[RARE] Terminating due to loop limit: {loop_count} >= {MAX_LOOP_COUNT}")
            return True
        elif current_calls >= MAX_API_CALLS:
            logger.info(f"[RARE] Terminating due to API call limit: {current_calls} >= {MAX_API_CALLS}")
            return True

        return False
    
    def get_pipeline_statistics(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        redundancy_mapping: Dict[str, RedundancyMapping],
        evaluation_items: List[RareEvaluationItem]
    ) -> Dict[str, int]:
        """Get pipeline statistics"""
        
        total_atomic = sum(len(units) for units in atomic_info_map.values())
        unique_count = sum(1 for r in redundancy_mapping.values() if r.redundant_items == ["unique"])
        redundant_count = len(redundancy_mapping) - unique_count
        
        return {
            "total_documents": len(atomic_info_map),
            "total_atomic_info": total_atomic,
            "unique_atomic_info": unique_count,
            "redundant_atomic_info": redundant_count,
            "generated_evaluation_items": len(evaluation_items),
            "total_api_calls": self.llm_client.get_total_api_calls(),
        }



    # =============================================================================
    # Multi-hop Question Generation Methods (HotPot-style Integration)
    # =============================================================================

    def generate_multihop_questions(
        self,
        gold_sentences: List[Dict],
        model: str = "gpt5_nano",
        max_workers: int = 128,
        num_candidates: int = 1
    ) -> Dict[str, Any]:
        """Generate multi-hop reasoning questions from gold sentences using RARE framework"""
        
        logger.info(f"[RARE MULTIHOP] Question Generation - Processing {len(gold_sentences)} gold sentences")
        
        # Format input for question generation
        input_sentences = self._format_gold_sentences_for_prompt(gold_sentences)
        
        try:
            # Generate question(s) using RARE prompt system
            prompt = generate_prompt(
                PromptType.GENERATE_MULTIHOP_QUESTION,
                num_candidates=num_candidates,
                input_sentences=input_sentences
            )
            
            # Call LLM for question generation
            response = self.llm_client.call_api(prompt, model, validate_json=True)
            result = clean_and_parse_json(response)
            
            if not result:
                logger.warning(f"[RARE MULTIHOP] Failed to parse question generation response")
                return None
            
            logger.info(f"[RARE MULTIHOP] Successfully generated question")
            return result
            
        except Exception as e:
            logger.error(f"[RARE MULTIHOP] Question generation error: {str(e)}")
            return None

    def validate_multihop_question(
        self,
        question: str,
        answer: str,
        gold_sentences: List[Dict],
        model: str = "gpt5_nano",
        max_workers: int = 64
    ) -> Dict[str, Any]:
        """Validate multi-hop reasoning question using RARE framework"""
        
        logger.info(f"[RARE MULTIHOP] Question Validation - Evaluating generated question")
        
        # Format sentences for validation prompt
        sentences_text = self._format_gold_sentences_for_validation(gold_sentences)
        
        try:
            # NOTE: VALIDATE_MULTIHOP_QUESTION removed - validation now integrated into generation
            # This method is deprecated
            logger.warning(f"[RARE MULTIHOP] validate_multihop_question is deprecated - validation integrated into generation")
            return None
            
        except Exception as e:
            logger.error(f"[RARE MULTIHOP] Question validation error: {str(e)}")
            return None

    def generate_and_validate_multihop_question(
        self,
        gold_sentences: List[Dict],
        model: str = "gpt5_nano",
        max_workers: int = 64
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate and validate multi-hop question in one pipeline call"""
        
        logger.info(f"[RARE MULTIHOP] Full Pipeline - Generate and validate question")
        
        # Step 1: Generate question
        generated_result = self.generate_multihop_questions(
            gold_sentences=gold_sentences,
            model=model,
            max_workers=max_workers
        )
        
        if not generated_result:
            return None, None
        
        # Step 2: Validate generated question
        validation_result = self.validate_multihop_question(
            question=generated_result.get('generated_question', ''),
            answer=generated_result.get('generated_answer', ''),
            gold_sentences=gold_sentences,
            model=model,
            max_workers=max_workers
        )
        
        return generated_result, validation_result

    def _format_gold_sentences_for_prompt(self, gold_sentences: List[Dict]) -> str:
        """Format gold sentences for question generation prompt"""
        sentence_text = ""
        for i, sent_info in enumerate(gold_sentences, 1):
            title = sent_info.get('title', 'Unknown')
            sentence = sent_info.get('sentence', '')
            sentence_text += f"Sentence {i} (Source: {title}):\n{sentence}\n\n"
        return sentence_text.strip()

    def _format_gold_sentences_for_validation(self, gold_sentences: List[Dict]) -> str:
        """Format gold sentences for question validation prompt"""
        sentences_text = ""
        for i, sent_info in enumerate(gold_sentences, 1):
            sentence = sent_info.get('sentence', '')
            sentences_text += f"{i}. {sentence}\n"
        return sentences_text.strip()

    def validate_multihop_questions_batch(
        self,
        gold_sentences: List[Dict],
        questions: List[Dict],
        model: str = "gpt5_nano",
        max_workers: int = 64
    ) -> List[Dict]:
        """Validate multiple questions and return scores for each"""
        
        logger.info(f"[RARE MULTIHOP] Batch Validation - Evaluating {len(questions)} questions")
        
        # Format input for validation
        input_sentences = self._format_gold_sentences_for_prompt(gold_sentences)
        questions_text = self._format_questions_for_validation(questions)
        
        try:
            # Validate questions using RARE prompt system
            prompt = generate_prompt(
                PromptType.VALIDATE_MULTIHOP_QUESTIONS,
                input_sentences=input_sentences,
                questions_list=questions_text,
                num_questions=len(questions)
            )
            
            # Call LLM for question validation
            response = self.llm_client.call_api(prompt, model, validate_json=True)
            result = clean_and_parse_json(response)
            
            if not result:
                logger.warning(f"[RARE MULTIHOP] Failed to parse validation response")
                return []
            
            logger.info(f"[RARE MULTIHOP] Successfully validated {len(questions)} questions")
            return result if isinstance(result, list) else []
            
        except Exception as e:
            logger.error(f"[RARE MULTIHOP] Batch validation error: {str(e)}")
            return []

    def step3_5_validate_atomic_info_separate(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        chunks: List[dict],
        language: str,
        model: str,
        max_workers: int = 64,
        fewshot_examples_by_chunk: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[AtomicInfo]]:
        """Step 3.5: Validate atomic information Separate"""
        
        total_input_chunks = len(atomic_info_map)
        total_atomic_units = sum(len(units) for units in atomic_info_map.values())
        logger.info(f"Step 3.5: Separate Validation - {total_atomic_units} units")
        
        validated_atomic_info_map = {}
        
        # Create chunk lookup for document content
        chunk_lookup = {}
        for chunk in chunks:
            doc_id = self._generate_doc_id(chunk, 0)  # We'll match by file/content
            chunk_lookup[chunk.get("file_name", "unknown")] = chunk.get("content", "")
        
        # Group atomic info by chunk for bulk validation
        chunk_validation_tasks = {}
        total_filtered_count = 0
        
        for chunk_id, atomic_list in atomic_info_map.items():
            # Get document content and title for this chunk
            file_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            doc_content = chunk_lookup.get(file_name, "")
            doc_title = file_name  # Use actual file name as title
            
            # Use all atomic info (keyword filtering removed)
            filtered_atomic_list = atomic_list
        
            if filtered_atomic_list:  # Only add if there are atomic info to validate
                chunk_validation_tasks[chunk_id] = (filtered_atomic_list, doc_content, doc_title, language, model)
        
        logger.info(f"Step 3.5: After filtering - {total_filtered_count} items")
        logger.info(f"[DEBUG Separate] Input chunks: {total_input_chunks}, Validation tasks: {len(chunk_validation_tasks)}")
        
        # Process bulk validation with progress bar
        valid_atomic_info = []  # Threshold-filtered atomic info
        all_atomic_info = []    # All atomic info (for dual comparison mode)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit bulk validation tasks (one per chunk)
            future_to_chunk = {
                executor.submit(
                    self._validate_separate_chunk_atomic_info,
                    atomic_list,
                    doc_content,
                    doc_title,
                    language,
                    model,
                    chunk_id,
                    (fewshot_examples_by_chunk.get(chunk_id) if fewshot_examples_by_chunk else None)
                ): chunk_id
                for chunk_id, (atomic_list, doc_content, doc_title, language, model) in chunk_validation_tasks.items()
            }
            
            # Process results with progress bar
            with tqdm(total=len(chunk_validation_tasks), desc="Separate validating chunks", unit="chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    
                    try:
                        validated_atomic_list = future.result()  # Returns List[AtomicInfo] with updated scores
                        
                        # DEBUG: Track chunk processing
                        logger.info(f"[DEBUG Separate] Chunk {chunk_id}: received {len(validated_atomic_list)} atomic items")
                        
                        # Count scores by availability
                        zero_confidence_count = 0
                        valid_threshold_count = 0
                        
                        # Collect ALL atomic info (for dual comparison mode) - EXCLUDE zero-scored items
                        for validated_atomic_info in validated_atomic_list:
                            # Only include items that were actually validated (non-zero confidence)
                            if validated_atomic_info.overall_confidence > 0.0:
                                all_atomic_info.append((validated_atomic_info, chunk_id))
                            else:
                                zero_confidence_count += 1
                        
                        # Filter valid atomic info based on overall_confidence threshold
                        for validated_atomic_info in validated_atomic_list:
                            # logger.debug(f"[DEBUG COMBINED] Item: '{validated_atomic_info.content[:30]}...' - confidence: {validated_atomic_info.overall_confidence:.3f}, threshold: {VALIDITY_CONFIDENCE_THRESHOLD}")
                            
                            if validated_atomic_info.overall_confidence >= VALIDITY_CONFIDENCE_THRESHOLD:
                                valid_atomic_info.append((validated_atomic_info, chunk_id))
                                valid_threshold_count += 1
                                logger.debug(f"[Step 3.5] Valid: {validated_atomic_info.content[:50]}... " +
                                           f"(overall: {validated_atomic_info.overall_confidence:.2f}, " +
                                           f"v: {validated_atomic_info.validity_score:.2f}, " +
                                           f"c: {validated_atomic_info.completeness_score:.2f}, " +
                                           f"s: {validated_atomic_info.specificity_score:.2f})", 
                                           f"cl: {validated_atomic_info.clarity_score:.2f}, " +
                                           f"q: {validated_atomic_info.questionability_score:.2f})"
                                           ) 
                            else:
                                logger.debug(f"[Step 3.5] Invalid: {validated_atomic_info.content[:50]}... " +
                                           f"(overall: {validated_atomic_info.overall_confidence:.2f})")
                        
                        # DEBUG: Chunk summary
                        logger.info(f"[DEBUG Separate] Chunk {chunk_id}: {zero_confidence_count} zero-confidence, {valid_threshold_count} above threshold")
                        
                    except Exception as e:
                        logger.error(f"[Step 3.5] Error separate validating chunk {chunk_id}: {e}")
                        logger.info(f"[DEBUG Separate] Chunk {chunk_id}: FAILED validation")
                    
                    # Update progress bar with detailed metrics
                    current_cost = self.llm_client.get_cost_summary().total_cost_usd
                    avg_overall = sum(item[0].overall_confidence for item in valid_atomic_info) / len(valid_atomic_info) if valid_atomic_info else 0.0
                    processed_chunks = pbar.n + 1
                    pbar.set_postfix({
                        'Cost': f'${current_cost:.4f}',
                        'Valid_Items': len(valid_atomic_info),
                        'Avg_Score': f'{avg_overall:.2f}',
                        'Chunks': f'{processed_chunks}/{len(chunk_validation_tasks)}'
                        })
                    pbar.update(1)
                        
        # DEBUG: Final summary statistics for Combined validation
        logger.info(f"[DEBUG Separate] FINAL SUMMARY:")
        logger.info(f"[DEBUG Separate] Input chunks: {total_input_chunks}")
        logger.info(f"[DEBUG Separate] Validation tasks: {len(chunk_validation_tasks)}")
        logger.info(f"[DEBUG Separate] All atomic items (confidence > 0): {len(all_atomic_info)}")
        logger.info(f"[DEBUG Separate] Valid atomic items (>= {VALIDITY_CONFIDENCE_THRESHOLD}): {len(valid_atomic_info)}")
        
        # Group validated atomic info back by chunk_id (threshold filtered)
        validated_atomic_info_map = {}
        for atomic_info, chunk_id in valid_atomic_info:
            if chunk_id not in validated_atomic_info_map:
                validated_atomic_info_map[chunk_id] = []
            validated_atomic_info_map[chunk_id].append(atomic_info)
        
        # Group ALL atomic info back by chunk_id (for dual comparison)
        all_atomic_info_map = {}
        for atomic_info, chunk_id in all_atomic_info:
            if chunk_id not in all_atomic_info_map:
                all_atomic_info_map[chunk_id] = []
            all_atomic_info_map[chunk_id].append(atomic_info)
        
        # DEBUG: Final chunk count after grouping
        logger.info(f"[DEBUG Separate] FINAL OUTPUT CHUNKS:")
        logger.info(f"[DEBUG Separate] Threshold-filtered chunks: {len(validated_atomic_info_map)}")
        logger.info(f"[DEBUG Separate] All chunks (confidence > 0): {len(all_atomic_info_map)}")
        
        # Save validated result WITH ALL validation scores
        validated_dict = {}
        for chunk_id, atomic_list in validated_atomic_info_map.items():
            validated_dict[chunk_id] = [
                {
                    "content": ai.content,
                    "chunk_id": ai.chunk_id,
                    "atomic_info_id": ai.atomic_info_id,
                    "validity_score": ai.validity_score,
                    "completeness_score": ai.completeness_score,
                    "specificity_score": ai.specificity_score,
                    "clarity_score": ai.clarity_score,
                    "questionability_score": ai.questionability_score,
                    "overall_confidence": ai.overall_confidence,
                    "validation_reasoning": ai.validation_reasoning
                }
                for ai in atomic_list
            ]
        
        # Calculate and add chunk-based rankings for both threshold-filtered and all data
        # 1. Threshold-filtered rankings
        flat_atomic_list = []
        for atomic_list in validated_atomic_info_map.values():
            flat_atomic_list.extend(atomic_list)
        
        chunk_rankings_filtered = generate_chunk_based_rankings_by_rank_average(flat_atomic_list)
        
        # 2. All data rankings (for dual comparison mode)
        all_flat_atomic_list = []
        for atomic_list in all_atomic_info_map.values():
            all_flat_atomic_list.extend(atomic_list)
        
        chunk_rankings_all = generate_chunk_based_rankings_by_rank_average(all_flat_atomic_list)
        
        # Save all atomic info data structure (for dual comparison)
        all_atomic_dict = {}
        for chunk_id, atomic_list in all_atomic_info_map.items():
            all_atomic_dict[chunk_id] = [
                {
                    "content": ai.content,
                    "chunk_id": ai.chunk_id,
                    "atomic_info_id": ai.atomic_info_id,
                    "validity_score": ai.validity_score,
                    "completeness_score": ai.completeness_score,
                    "specificity_score": ai.specificity_score,
                    "overall_confidence": ai.overall_confidence,
                    "validation_reasoning": ai.validation_reasoning
                }
                for ai in atomic_list
            ]
        
        # Create simple ranking by chunk (like GOLD ranking) - sorted by chunk ID
        validation_ranking_by_chunk = {}
        for chunk_id in sorted(chunk_rankings_all.keys()):
            validation_ranking_by_chunk[chunk_id] = chunk_rankings_all[chunk_id]
        
        # Add ranking information to save data (dual comparison mode)
        validated_dict_with_rankings = {
            "validation_method": "combined",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "validation_ranking_by_chunk": validation_ranking_by_chunk,  # Simple ranking like GOLD
            "comparison_modes": {
                "threshold_filtered": {
                    "atomic_info_by_chunk": validated_dict,
                    "chunk_rankings": chunk_rankings_filtered,
                    "threshold": VALIDITY_CONFIDENCE_THRESHOLD,
                    "total_items": len(flat_atomic_list)
                },
                "all_items": {
                    "atomic_info_by_chunk": all_atomic_dict,
                    "chunk_rankings": chunk_rankings_all,
                    "threshold": "none",
                    "total_items": len(all_flat_atomic_list)
                }
            },
            "ranking_metadata": {
                "ranking_method": "overall_confidence_desc",
                "total_chunks": len(chunk_rankings_all)
            }
        }
        
        output_file = self.output_dir / "step3_5_validated_atomic_info_separate.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_dict_with_rankings, f, ensure_ascii=False, indent=2)
        
        total_validated = sum(len(units) for units in validated_atomic_info_map.values())
        logger.info(f"Step 3.5: Completed - {total_validated} validated")
        
        # Return both filtered and all data for SQuAD evaluation compatibility
        return validated_dict_with_rankings

    def validate_multihop_questions_batch_separate(
        self,
        gold_sentences: List[Dict],
        questions: List[Dict],
        model: str = "gpt5_nano",
        max_workers: int = 64
    ) -> List[Dict]:
        """Validate multiple questions using separate validation calls (4 calls total"""
        
        logger.info(f"[RARE MULTIHOP] Separate Validation - Evaluating {len(questions)} questions with 4 separate calls")
        
        # Format input for validation
        input_sentences = self._format_gold_sentences_for_prompt(gold_sentences)
        questions_text = self._format_questions_for_validation(questions)
        
        try:
            # Call 1: Connectivity validation
            connectivity_prompt = generate_prompt(
                PromptType.VALIDATE_MULTIHOP_QUESTIONS_CONNECTIVITY_SEPARATE,
                input_sentences=input_sentences,
                questions_list=questions_text,
                num_questions=len(questions)
            )
            connectivity_response = self.llm_client.call_api_with_score_validation(
                prompt=connectivity_prompt,
                validation_type='separate_connectivity',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            connectivity_results = clean_and_parse_json(connectivity_response)
            connectivity_results = connectivity_results if isinstance(connectivity_results, list) else []
            
            # Call 2: Fluency validation
            fluency_prompt = generate_prompt(
                PromptType.VALIDATE_MULTIHOP_QUESTIONS_FLUENCY_SEPARATE,
                input_sentences=input_sentences,
                questions_list=questions_text,
                num_questions=len(questions)
            )
            fluency_response = self.llm_client.call_api_with_score_validation(
                prompt=fluency_prompt,
                validation_type='separate_fluency',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            fluency_results = clean_and_parse_json(fluency_response)
            fluency_results = fluency_results if isinstance(fluency_results, list) else []
            
            # Call 3: Essentiality validation
            essentiality_prompt = generate_prompt(
                PromptType.VALIDATE_MULTIHOP_QUESTIONS_ESSENTIALITY_SEPARATE,
                input_sentences=input_sentences,
                questions_list=questions_text,
                num_questions=len(questions)
            )
            essentiality_response = self.llm_client.call_api_with_score_validation(
                prompt=essentiality_prompt,
                validation_type='separate_essentiality',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            essentiality_results = clean_and_parse_json(essentiality_response)
            essentiality_results = essentiality_results if isinstance(essentiality_results, list) else []
            
            # Call 4: Validity validation
            validity_prompt = generate_prompt(
                PromptType.VALIDATE_MULTIHOP_QUESTIONS_VALIDITY_SEPARATE,
                input_sentences=input_sentences,
                questions_list=questions_text,
                num_questions=len(questions)
            )
            validity_response = self.llm_client.call_api_with_score_validation(
                prompt=validity_prompt,
                validation_type='separate_validity',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            validity_results = clean_and_parse_json(validity_response)
            validity_results = validity_results if isinstance(validity_results, list) else []
            
            # Combine results from all 4 validations for each question
            combined_results = []
            for i in range(len(questions)):
                question_result = {
                    'candidate_id': questions[i].get('candidate_id', i + 1),
                    'reasoning': f"Separate validation - 4 independent dimension evaluations"
                }
                
                # Extract scores from each validation result                   
                if i < len(connectivity_results):
                    question_result['connectivity_score'] = connectivity_results[i].get('connectivity_score', 0.5)
                else:
                    question_result['connectivity_score'] = 0.5
                    
                if i < len(fluency_results):
                    question_result['fluency_score'] = fluency_results[i].get('fluency_score', 0.5)
                else:
                    question_result['fluency_score'] = 0.5
                    
                if i < len(essentiality_results):
                    question_result['essentiality_score'] = essentiality_results[i].get('essentiality_score', 0.5)
                else:
                    question_result['essentiality_score'] = 0.5
                    
                if i < len(validity_results):
                    question_result['validity_score'] = validity_results[i].get('validity_score', 0.5)
                else:
                    question_result['validity_score'] = 0.5
                
                combined_results.append(question_result)
            
            logger.info(f"[RARE MULTIHOP] Successfully validated {len(questions)} questions with separate calls")
            return combined_results
            
        except Exception as e:
            logger.error(f"[RARE MULTIHOP] Separate validation error: {str(e)}")
            return []
    
    def filter_questions_logical_consistency(
        self,
        gold_sentences: List[Dict],
        questions: List[Dict],
        model: str = "gpt5_nano"
    ) -> List[Dict]:
        """Filter questions based on logical consistency using 4 separate validation calls."""
        
        logger.debug(f"[RARE MULTIHOP] Logical consistency filtering: {len(questions)} questions, 4 separate calls")
        
        questions_text = self._format_questions_for_validation(questions)
        
        try:
            # Call 1: Context Assumption Check
            context_prompt = generate_prompt(
                PromptType.FILTER_CONTEXT_ASSUMPTION,
                questions_list=questions_text,
                num_questions=len(questions)
            )
            context_response = self.llm_client.call_api_with_score_validation(
                prompt=context_prompt,
                validation_type='filter_context_assumption',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            context_results = clean_and_parse_json(context_response)
            context_results = context_results if isinstance(context_results, list) else []
            
            # Call 2: Circular Definition Check
            circular_prompt = generate_prompt(
                PromptType.FILTER_CIRCULAR_DEFINITION,
                questions_list=questions_text,
                input_sentences=self._format_gold_sentences_for_filtering(gold_sentences),
                num_questions=len(questions)
            )
            circular_response = self.llm_client.call_api_with_score_validation(
                prompt=circular_prompt,
                validation_type='filter_circular_definition',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            circular_results = clean_and_parse_json(circular_response)
            circular_results = circular_results if isinstance(circular_results, list) else []
            
            # Call 3: Information Completeness Check
            completeness_prompt = generate_prompt(
                PromptType.FILTER_INFORMATION_COMPLETENESS,
                questions_list=questions_text,
                input_sentences=self._format_gold_sentences_for_filtering(gold_sentences),
                num_questions=len(questions)
            )
            completeness_response = self.llm_client.call_api_with_score_validation(
                prompt=completeness_prompt,
                validation_type='filter_information_completeness',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            completeness_results = clean_and_parse_json(completeness_response)
            completeness_results = completeness_results if isinstance(completeness_results, list) else []
            
            # Call 4: Question Ambiguity Check
            ambiguity_prompt = generate_prompt(
                PromptType.FILTER_QUESTION_AMBIGUITY,
                questions_list=questions_text,
                num_questions=len(questions)
            )
            ambiguity_response = self.llm_client.call_api_with_score_validation(
                prompt=ambiguity_prompt,
                validation_type='filter_question_ambiguity',
                expected_count=len(questions),
                model=model,
                max_retries=3
            )
            ambiguity_results = clean_and_parse_json(ambiguity_response)
            ambiguity_results = ambiguity_results if isinstance(ambiguity_results, list) else []
            
            # Filter questions: ALL 4 dimensions must pass
            valid_questions = []
            filtered_questions_by_type = {
                'context_assumption_errors': [],
                'circular_definition_errors': [],
                'information_equivalence_errors': [],
                'question_ambiguity_errors': []
            }
            filtering_stats = {
                'context_assumption_errors': 0,
                'circular_definition_errors': 0,
                'information_equivalence_errors': 0,
                'question_ambiguity_errors': 0
            }
            
            for i, question in enumerate(questions):
                # Get results for this question from all 4 checks (case-insensitive)
                context_error = str(context_results[i].get('contextual_independence_check', 'fail')).lower() == 'fail' if i < len(context_results) else True
                circular_error = str(
                    circular_results[i].get('answer_exclusion_check', 'fail')
                ).lower() == 'fail' if i < len(circular_results) else True
                completeness_error = str(
                    completeness_results[i].get('information_equivalence_check', 'fail')
                ).lower() == 'fail' if i < len(completeness_results) else True
                ambiguity_error = str(ambiguity_results[i].get('question_ambiguity_check', 'fail')).lower() == 'fail' if i < len(ambiguity_results) else True
                
                # Track error statistics and save failed questions by type
                if context_error:
                    filtering_stats['context_assumption_errors'] += 1
                    filtered_questions_by_type['context_assumption_errors'].append({
                        'question': question,
                        'reasoning': context_results[i].get('reasoning', 'No reasoning provided') if i < len(context_results) else 'API failure'
                    })
                if circular_error:
                    filtering_stats['circular_definition_errors'] += 1
                    filtered_questions_by_type['circular_definition_errors'].append({
                        'question': question,
                        'reasoning': circular_results[i].get('reasoning', 'No reasoning provided') if i < len(circular_results) else 'API failure'
                    })
                if completeness_error:
                    filtering_stats['information_equivalence_errors'] += 1
                    filtered_questions_by_type['information_equivalence_errors'].append({
                        'question': question,
                        'reasoning': completeness_results[i].get('reasoning', 'No reasoning provided') if i < len(completeness_results) else 'API failure'
                    })
                if ambiguity_error:
                    filtering_stats['question_ambiguity_errors'] += 1
                    filtered_questions_by_type['question_ambiguity_errors'].append({
                        'question': question,
                        'reasoning': ambiguity_results[i].get('reasoning', 'No reasoning provided') if i < len(ambiguity_results) else 'API failure'
                    })
                
                # Pass only if ALL 4 dimensions are clean (no errors)
                if not (context_error or circular_error or completeness_error or ambiguity_error):
                    valid_questions.append(question)
                else:
                    # Log detailed reasoning for filtered questions (debug level)
                    candidate_id = question.get('candidate_id', i+1)
                    error_details = []
                    
                    if context_error and i < len(context_results):
                        error_details.append(f"Context: {context_results[i].get('reasoning', 'No reasoning provided')[:100]}...")
                    if circular_error and i < len(circular_results):
                        error_details.append(f"Circular: {circular_results[i].get('reasoning', 'No reasoning provided')[:100]}...")
                    if completeness_error and i < len(completeness_results):
                        error_details.append(f"Information Equivalence: {completeness_results[i].get('reasoning', 'No reasoning provided')[:100]}...")
                    if ambiguity_error and i < len(ambiguity_results):
                        error_details.append(f"Ambiguity: {ambiguity_results[i].get('reasoning', 'No reasoning provided')[:100]}...")
                    
                    logger.debug(f"[RARE MULTIHOP] Filtered Q{candidate_id}: {' | '.join(error_details)}")
            
            filtered_count = len(questions) - len(valid_questions)
            logger.info(f"[RARE MULTIHOP] Filtering complete: {len(valid_questions)}/{len(questions)} passed")
            
            if filtered_count > 0:
                logger.info(f"[RARE MULTIHOP] Errors - Context:{filtering_stats['context_assumption_errors']} "
                          f"Conversion:{filtering_stats['circular_definition_errors']} "
                          f"InformationEquivalence:{filtering_stats['information_equivalence_errors']} "
                          f"Ambiguity:{filtering_stats['question_ambiguity_errors']}")
            
            # Return valid questions, detailed statistics, and filtered questions by type
            return {
                'valid_questions': valid_questions,
                'detailed_stats': filtering_stats,
                'filtered_questions_by_type': filtered_questions_by_type
            }
            
        except Exception as e:
            logger.error(f"[RARE MULTIHOP] Logical filtering error: {str(e)}")
            # Return in consistent format even on error
            return {
                'valid_questions': questions,
                'detailed_stats': {
                    'context_assumption_errors': 0,
                    'circular_definition_errors': 0,
                    'information_completeness_errors': 0,
                    'question_ambiguity_errors': 0
                },
                'filtered_questions_by_type': {
                    'context_assumption_errors': [],
                    'circular_definition_errors': [],
                    'information_completeness_errors': [],
                    'question_ambiguity_errors': []
                }
            }

    def _format_questions_for_validation(self, questions: List[Dict]) -> str:
        """Format questions for validation prompt"""
        questions_text = ""
        for i, q in enumerate(questions, 1):
            question = q.get('generated_question', '')
            answer = q.get('generated_answer', '')
            questions_text += f"Question {i} (ID: {q.get('candidate_id', i)}):\n{question}\nAnswer: {answer}\n\n"
        return questions_text.strip()
        
    def _format_gold_sentences_for_filtering(self, gold_sentences: List[Dict]) -> str:
        """Format gold sentences for information completeness filtering prompt"""
        sentences_text = ""
        for i, sentence in enumerate(gold_sentences, 1):
            # Include title and content for complete context
            title = sentence.get('title', 'Unknown')
            content = sentence.get('content', '')
            sentences_text += f"IDX {i} - ({title}): {content}\n"
        return sentences_text.strip()

    def filter_atomic_info_logical_consistency(
        self,
        atomic_info_list: List[AtomicInfo],
        model: str = "gpt5_nano",
        max_workers: int = 128
    ) -> Dict[str, any]:
        """Filter atomic information based on information completeness using chunk-based batch processing."""
        
        logger.info(f"[RARE STEP 4] Atomic info completeness filtering: {len(atomic_info_list)} atomic units, chunk-based batch processing, {max_workers} workers")
        
        # Group atomic info by chunk_id for efficient batch processing
        chunk_atomic_map = {}
        for atomic_info in atomic_info_list:
            chunk_id = atomic_info.chunk_id
            if chunk_id not in chunk_atomic_map:
                chunk_atomic_map[chunk_id] = []
            chunk_atomic_map[chunk_id].append(atomic_info)
        
        logger.info(f"[RARE STEP 4] Grouped {len(atomic_info_list)} atomic units into {len(chunk_atomic_map)} chunks")
        
        # Process chunks in parallel using ThreadPoolExecutor
        valid_atomic_info = []
        filtered_atomic_info_by_type = {
            'information_completeness_errors': []
        }
        filtering_stats = {
            'information_completeness_errors': 0
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit filtering tasks for each chunk
            future_to_chunk = {
                executor.submit(self._filter_chunk_atomic_info_logical_consistency, chunk_id, chunk_atomic_list, model): chunk_id 
                for chunk_id, chunk_atomic_list in chunk_atomic_map.items()
            }
            
            # Process results with progress bar
            with tqdm(total=len(chunk_atomic_map), desc="Completeness filtering chunks", unit="chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    
                    try:
                        chunk_filtering_results = future.result()  # Returns list of filtering results for atomic info in this chunk
                        
                        # Process each atomic info result in this chunk
                        for atomic_info, filtering_result in zip(chunk_atomic_map[chunk_id], chunk_filtering_results):
                            # Check completeness dimension
                            completeness_error = filtering_result.get('information_completeness_error', True)
                            
                            # Count errors
                            if completeness_error:
                                filtering_stats['information_completeness_errors'] += 1
                                filtered_atomic_info_by_type['information_completeness_errors'].append(atomic_info)
                            else:
                                # Only include if completeness check passes
                                valid_atomic_info.append(atomic_info)
                        
                        # Update progress bar with current stats
                        current_cost = self.llm_client.get_total_cost_usd()
                        pass_count = len(valid_atomic_info)
                        
                        pbar.set_postfix(Cost=f"${current_cost:.4f}", Passed=pass_count)
                        
                    except Exception as e:
                        logger.error(f"[RARE STEP 4] Error filtering chunk {chunk_id}: {str(e)}")
                        # On error, assume completeness check fails for all atomic info in this chunk
                        chunk_size = len(chunk_atomic_map[chunk_id])
                        filtering_stats['information_completeness_errors'] += chunk_size
                        
                    pbar.update(1)
        
        total_passed = len(valid_atomic_info)
        total_original = len(atomic_info_list)
        pass_rate = (total_passed / total_original * 100) if total_original > 0 else 0
        
        logger.info(f"[RARE STEP 4] Completeness filtering results: {total_passed}/{total_original} passed ({pass_rate:.1f}%) - "
                      f"Incomplete:{filtering_stats['information_completeness_errors']}")
        
        # Return valid atomic info, detailed statistics, and filtered atomic info by type
        return {
            'valid_atomic_info': valid_atomic_info,
            'detailed_stats': filtering_stats,
            'filtered_atomic_info_by_type': filtered_atomic_info_by_type
        }

    def _filter_chunk_atomic_info_logical_consistency(
        self,
        chunk_id: str,
        atomic_info_list: List[AtomicInfo],
        model: str = "gpt5_nano"
    ) -> List[Dict[str, bool]]:
        """Filter chunk's atomic information using information completeness check in batch."""
        
        # Format atomic info list for filtering (multiple items)
        atomic_info_text = self._format_atomic_info_for_filtering(atomic_info_list)
        num_atomic = len(atomic_info_list)
        
        try:
            # Information Completeness Check (single call)
            completeness_prompt = generate_prompt(
                PromptType.FILTER_ATOMIC_INFORMATION_COMPLETENESS,
                atomic_info_list=atomic_info_text,
                num_atomic=num_atomic
            )
            completeness_response = self.llm_client.call_api_with_score_validation(
                prompt=completeness_prompt,
                validation_type='filter_atomic_information_completeness',
                expected_count=num_atomic,
                model=model,
                max_retries=3
            )
            completeness_results = clean_and_parse_json(completeness_response)
            completeness_results = completeness_results if isinstance(completeness_results, list) else []
            
            # Process results for each atomic info
            filtering_results = []
            for i in range(num_atomic):
                # Robust boolean parsing: handle "true"/"True"/true and "false"/"False"/false
                error_value = completeness_results[i].get('has_information_completeness_error', True) if i < len(completeness_results) else True
                if isinstance(error_value, str):
                    completeness_error = error_value.lower() == 'true'
                else:
                    completeness_error = bool(error_value)
                
                filtering_results.append({
                    'information_completeness_error': completeness_error
                })
            
            return filtering_results
            
        except Exception as e:
            logger.error(f"[RARE STEP 4] Error in chunk atomic info filtering {chunk_id}: {str(e)}")
            # On error, assume completeness check fails for all atomic info
            return [{
                'information_completeness_error': True
            } for _ in range(num_atomic)]

    def _format_atomic_info_for_filtering(self, atomic_info_list: List[AtomicInfo]) -> str:
        """Format atomic information for completeness filtering prompts"""
        atomic_info_text = ""
        for i, atomic_info in enumerate(atomic_info_list, 1):
            content = atomic_info.content
            atomic_id = atomic_info.atomic_info_id
            atomic_info_text += f"Atomic ID {i} ({atomic_id}):\n{content}\n\n"
        return atomic_info_text.strip()
    
    # =============================================================================
    # Question Paraphrasing and Answerability Methods
    # =============================================================================
    
    def paraphrase_questions_batch(
        self,
        question_data: List[Dict[str, Any]],
        chunk_data_dict: Dict[str, Any] = None,
        model: str = "gpt5_nano"
    ) -> List[Dict[str, Any]]:
        """Paraphrase multiple questions to increase diversity while maintaining meaning."""
        
        try:
            # Format questions and answers for paraphrasing (with individual titles per question)
            questions_and_answers = ""
            all_used_titles = set()  # Track all titles used across questions
            
            for i, q_data in enumerate(question_data, 1):
                question = q_data.get('generated_question', '')
                answer = q_data.get('generated_answer', '')
                
                # Get titles directly from question data (provided by step7)
                question_titles = q_data.get('question_titles', [])
                all_used_titles.update(question_titles)
                
                questions_and_answers += f"Question {i}: {question}\n"
                questions_and_answers += f"Answer {i}: {answer}\n"
                questions_and_answers += f"Doc Titles {i}: {sorted(question_titles)}\n\n"
            
            # Format all used document titles for reference
            document_titles = ""
            for i, title in enumerate(sorted(all_used_titles), 1):
                document_titles += f"Document {i}: {title}\n"
            
            # Generate paraphrasing prompt (no context needed)
            paraphrase_prompt = generate_prompt(
                PromptType.PARAPHRASE_QUESTIONS,
                questions_and_answers=questions_and_answers.strip(),
                document_titles=document_titles.strip(),
                num_questions=len(question_data)
            )
            
            # Call LLM API
            response = self.llm_client.call_api(
                paraphrase_prompt,
                model=model,
                validate_json=True
            )
            
            result = clean_and_parse_json(response)
            
            if result and isinstance(result, list) and len(result) == len(question_data):
                return result
            else:
                logger.warning(f"[Paraphrase] Invalid or incomplete paraphrase results returned")
                return None
                
        except Exception as e:
            logger.error(f"[Paraphrase] Error paraphrasing questions: {str(e)}")
            return None
    
    def check_question_answerability(
        self,
        questions_with_chunks: List[Dict[str, Any]],
        chunk_data_dict: Dict[str, Any] = None,
        model: str = "gpt5_nano"
    ) -> List[Dict[str, Any]]:
        """Check if questions can be answered using their corresponding chunk content with titles."""
        
        try:
            # Format questions list
            questions_list = ""
            for i, q_data in enumerate(questions_with_chunks, 1):
                question = q_data.get('question', '')
                questions_list += f"Question {i}: {question}\n"
            
            # Format chunks with titles for each question
            chunks_with_titles = ""
            
            # Collect all unique chunks used across all questions
            all_chunk_ids = set()
            for q_data in questions_with_chunks:
                chunk_ids = q_data.get('chunk_ids', [])
                all_chunk_ids.update(chunk_ids)
            
            # Create chunk index mapping
            chunk_index_map = {chunk_id: i+1 for i, chunk_id in enumerate(sorted(all_chunk_ids))}
            
            # Format all unique chunks
            for chunk_id in sorted(all_chunk_ids):
                chunk_index = chunk_index_map[chunk_id]
                
                # Get content from chunk_data_dict if available
                content = ""
                title = f"Document {chunk_index}"
                if chunk_data_dict and chunk_id in chunk_data_dict:
                    content = chunk_data_dict[chunk_id].get('content', '')
                    source_title = chunk_data_dict[chunk_id].get('source_title', chunk_id)
                    title = source_title
                
                chunks_with_titles += f"Title {chunk_index}: {title}\n"
                chunks_with_titles += f"Chunk {chunk_index}: {content}\n\n"
            
            # Add question-specific chunk mapping information
            chunks_with_titles += "\n# Question-Chunk Mapping:\n"
            for i, q_data in enumerate(questions_with_chunks, 1):
                chunk_ids = q_data.get('chunk_ids', [])
                chunk_indices = [str(chunk_index_map[cid]) for cid in chunk_ids if cid in chunk_index_map]
                chunks_with_titles += f"Question {i} uses Chunks: {', '.join(chunk_indices)}\n"
            
            # Generate answerability check prompt
            answerability_prompt = generate_prompt(
                PromptType.VERIFY_QUESTION_ANSWERABILITY,
                questions_list=questions_list.strip(),
                chunks_with_titles=chunks_with_titles.strip(),
                num_questions=len(questions_with_chunks)
            )
            
            # Call LLM API
            response = self.llm_client.call_api(
                answerability_prompt,
                model=model,
                validate_json=True
            )
            
            result = clean_and_parse_json(response)
            
            if result and isinstance(result, list) and len(result) == len(questions_with_chunks):
                return result
            else:
                logger.warning(f"[Answerability] Invalid or incomplete answerability results returned")
                # Return fallback results
                return [{'reasoning': 'No valid response received', 'answerability_check': 'fail', 'generated_answer': ''} for _ in questions_with_chunks]
                
        except Exception as e:
            logger.error(f"[Answerability] Error checking answerability: {str(e)}")
            # Return fallback results for all questions
            return [{'reasoning': f'Error during analysis: {str(e)}', 'answerability_check': 'fail', 'generated_answer': ''} for _ in questions_with_chunks]

    # =============================================================================
    # Vanilla Validation Methods
    # =============================================================================
    
    def step3_5_validate_atomic_info_vanilla(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        chunks: List[dict],
        language: str,
        model: str,
        max_workers: int = 64,
        fewshot_examples_by_chunk: Dict[str, str] = None
    ) -> Dict[str, List[AtomicInfo]]:
        """Step 3.5: Validate atomic information quality using Vanilla approach (single overall quality score)"""
        
        total_input_chunks = len(atomic_info_map)
        total_atomic_units = sum(len(units) for units in atomic_info_map.values())
        logger.info(f"Step 3.5: Vanilla Validation - {total_atomic_units} units")
        
        validated_atomic_info_map = {}
        
        # Create chunk lookup for document content
        chunk_lookup = {}
        for chunk in chunks:
            doc_id = self._generate_doc_id(chunk, 0)  # We'll match by file/content
            chunk_lookup[chunk.get("file_name", "unknown")] = chunk.get("content", "")
        
        # Group atomic info by chunk for bulk validation
        chunk_validation_tasks = {}
        
        for chunk_id, atomic_list in atomic_info_map.items():
            # Get document content and title for this chunk
            file_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            doc_content = chunk_lookup.get(file_name, "")
            doc_title = file_name  # Use actual file name as title
            
            # Use all atomic info 
            filtered_atomic_list = atomic_list
            
            if filtered_atomic_list:  # Only add if there are atomic info to validate
                chunk_validation_tasks[chunk_id] = (filtered_atomic_list, doc_content, doc_title, language, model)
        
        logger.info(f"[DEBUG Vanilla] Input chunks: {total_input_chunks}, Validation tasks: {len(chunk_validation_tasks)}")
        
        # Process bulk validation with progress bar
        valid_atomic_info = []  # Threshold-filtered atomic info
        all_atomic_info = []    # All atomic info (for dual comparison mode)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit bulk validation tasks (one per chunk)
            future_to_chunk = {
                executor.submit(self._validate_bulk_atomic_info_vanilla, atomic_list, doc_content, doc_title, language, model, fewshot_examples_by_chunk.get(chunk_id) if fewshot_examples_by_chunk else None): 
                chunk_id for chunk_id, (atomic_list, doc_content, doc_title, language, model) in chunk_validation_tasks.items()
            }
            
            with tqdm(total=len(chunk_validation_tasks), desc="Validating atomic info (Vanilla)") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        validated_atomic_list = future.result()
                        validated_atomic_info_map[chunk_id] = validated_atomic_list
                        
                        # Update progress details
                        pbar.set_postfix({
                            'Cost': f'${self.llm_client.get_cost_summary().total_cost_usd:.4f}',
                            'Items': len(validated_atomic_list),
                            'Chunk': chunk_id.split('_')[-1] if '_' in chunk_id else chunk_id
                        })
                        pbar.update(1)
                        
                        # Filter based on threshold for statistics
                        for atomic_info in validated_atomic_list:
                            all_atomic_info.append(atomic_info)
                            # Use overall_quality_score as threshold instead of overall_confidence
                            if atomic_info.overall_quality_score >= VALIDITY_CONFIDENCE_THRESHOLD:
                                valid_atomic_info.append(atomic_info)
                            
                    except Exception as e:
                        logger.error(f"[Vanilla] Failed to validate atomic info for chunk {chunk_id}: {e}")
                        validated_atomic_info_map[chunk_id] = []
                        pbar.update(1)
        
        # Log validation statistics
        total_validated = len(all_atomic_info)
        total_valid = len(valid_atomic_info)
        logger.info(f"Step 3.5: Vanilla Results - {total_valid}/{total_validated} units passed threshold")
        logger.info(f"Step 3.5: Total validation cost: ${self.llm_client.get_cost_summary().total_cost_usd:.4f}")
        
        return validated_atomic_info_map
    
    def step3_5_validate_atomic_info_combined_ensemble(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        chunks: List[dict],
        language: str,
        model: str,
        max_workers: int = 64,
        ensemble_count: int = 5
    ) -> Dict[str, List[AtomicInfo]]:
        """Step 3.5: Combined Ensemble - Validate atomic information by running Combined 5 times for ensemble effect"""
        
        total_input_chunks = len(atomic_info_map)
        total_atomic_units = sum(len(units) for units in atomic_info_map.values())
        logger.info(f"Step 3.5: Combined Ensemble Validation - {total_atomic_units} units, {ensemble_count} runs")
        
        # Store all ensemble results
        all_ensemble_results = []
        
        # Create chunk lookup for document content
        chunk_lookup = {}
        for chunk in chunks:
            doc_id = self._generate_doc_id(chunk, 0)
            chunk_lookup[chunk.get("file_name", "unknown")] = chunk.get("content", "")
        
        # Run combined validation multiple times
        for run_idx in range(ensemble_count):
            logger.info(f"[Combined Ensemble] Run {run_idx + 1}/{ensemble_count}")
            
            # Group atomic info by chunk for bulk validation
            chunk_validation_tasks = {}
            
            for chunk_id, atomic_list in atomic_info_map.items():
                # Get document content and title for this chunk
                file_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
                doc_content = chunk_lookup.get(file_name, "")
                doc_title = file_name
                
                # Use all atomic info (keyword filtering removed)
                filtered_atomic_list = atomic_list
                
                if filtered_atomic_list:
                    chunk_validation_tasks[chunk_id] = (filtered_atomic_list, doc_content, doc_title, language, model)
            
            # Process bulk validation for this run
            run_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit bulk validation tasks (one per chunk)
                future_to_chunk = {
                    executor.submit(self._validate_bulk_atomic_info, atomic_list, doc_content, doc_title, language, model): 
                    chunk_id for chunk_id, (atomic_list, doc_content, doc_title, language, model) in chunk_validation_tasks.items()
                }
                
                # Process results with progress bar
                with tqdm(total=len(chunk_validation_tasks), desc=f"Combined Ensemble Run {run_idx + 1}", unit="chunks") as pbar:
                    for future in as_completed(future_to_chunk):
                        chunk_id = future_to_chunk[future]
                        
                        try:
                            validated_atomic_list = future.result()
                            
                            # Collect results for this run
                            for validated_atomic_info in validated_atomic_list:
                                if validated_atomic_info.overall_confidence > 0.0:
                                    run_results.append((validated_atomic_info, chunk_id, run_idx))
                                    
                        except Exception as e:
                            logger.error(f"[Combined Ensemble] Error validating chunk {chunk_id} in run {run_idx}: {e}")
                        
                        # Update progress bar
                        current_cost = self.llm_client.get_cost_summary().total_cost_usd
                        pbar.set_postfix({
                            'Cost': f'${current_cost:.4f}',
                            'Items': len(run_results),
                            'Run': f'{run_idx + 1}/{ensemble_count}'
                        })
                        pbar.update(1)
            
            all_ensemble_results.extend(run_results)
            logger.info(f"[Combined Ensemble] Run {run_idx + 1} completed: {len(run_results)} items")
        
        # Aggregate ensemble results - average scores for same atomic info
        ensemble_aggregated = {}  # chunk_id -> {content -> aggregated_atomic_info}
        
        for atomic_info, chunk_id, run_idx in all_ensemble_results:
            if chunk_id not in ensemble_aggregated:
                ensemble_aggregated[chunk_id] = {}
            
            content_key = atomic_info.content
            if content_key not in ensemble_aggregated[chunk_id]:
                # First occurrence - initialize with current scores
                ensemble_aggregated[chunk_id][content_key] = {
                    'atomic_info': AtomicInfo(
                        content=atomic_info.content,
                        chunk_id=chunk_id,
                        validity_score=atomic_info.validity_score,
                        completeness_score=atomic_info.completeness_score,
                        specificity_score=atomic_info.specificity_score,
                        clarity_score=atomic_info.clarity_score,
                        questionability_score=atomic_info.questionability_score,
                        overall_confidence=(atomic_info.validity_score + atomic_info.completeness_score + 
                                         atomic_info.specificity_score + atomic_info.clarity_score + 
                                         atomic_info.questionability_score) / 5.0
                    ),
                    'scores': [atomic_info],
                    'count': 1
                }
            else:
                # Subsequent occurrences - accumulate scores
                existing = ensemble_aggregated[chunk_id][content_key]
                existing['scores'].append(atomic_info)
                existing['count'] += 1
                
                # Update averaged scores
                total_validity = sum(item.validity_score for item in existing['scores'])
                total_completeness = sum(item.completeness_score for item in existing['scores'])
                total_specificity = sum(item.specificity_score for item in existing['scores'])
                total_clarity = sum(item.clarity_score for item in existing['scores'])
                total_questionability = sum(item.questionability_score for item in existing['scores'])
                
                count = existing['count']
                existing['atomic_info'].validity_score = total_validity / count
                existing['atomic_info'].completeness_score = total_completeness / count
                existing['atomic_info'].specificity_score = total_specificity / count
                existing['atomic_info'].clarity_score = total_clarity / count
                existing['atomic_info'].questionability_score = total_questionability / count
                existing['atomic_info'].overall_confidence = (
                    existing['atomic_info'].validity_score + existing['atomic_info'].completeness_score + 
                    existing['atomic_info'].specificity_score + existing['atomic_info'].clarity_score + 
                    existing['atomic_info'].questionability_score
                ) / 5.0
        
        # Convert to final validated atomic info map
        validated_atomic_info_map = {}
        for chunk_id, content_dict in ensemble_aggregated.items():
            validated_atomic_info_map[chunk_id] = []
            for content_key, aggregated_data in content_dict.items():
                atomic_info = aggregated_data['atomic_info']
                # Apply same threshold filtering as combined
                if atomic_info.overall_confidence >= VALIDITY_CONFIDENCE_THRESHOLD:
                    validated_atomic_info_map[chunk_id].append(atomic_info)
        
        # DEBUG: Final summary
        total_ensemble_items = len(all_ensemble_results)
        total_final_items = sum(len(items) for items in validated_atomic_info_map.values())
        logger.info(f"[Combined Ensemble] FINAL SUMMARY:")
        logger.info(f"[Combined Ensemble] Total ensemble results: {total_ensemble_items}")
        logger.info(f"[Combined Ensemble] Final aggregated items: {total_final_items}")
        logger.info(f"[Combined Ensemble] Average runs per item: {total_ensemble_items / total_final_items if total_final_items > 0 else 0:.1f}")
        
        return validated_atomic_info_map
    
    def step3_5_validate_atomic_info_vanilla_ensemble(
        self,
        atomic_info_map: Dict[str, List[AtomicInfo]],
        chunks: List[dict],
        language: str,
        model: str,
        max_workers: int = 64,
        ensemble_count: int = 5
    ) -> Dict[str, List[AtomicInfo]]:
        """Step 3.5: Vanilla Ensemble - Validate atomic information by running Vanilla 5 times for ensemble effect"""
        
        total_input_chunks = len(atomic_info_map)
        total_atomic_units = sum(len(units) for units in atomic_info_map.values())
        logger.info(f"Step 3.5: Vanilla Ensemble Validation - {total_atomic_units} units, {ensemble_count} runs")
        
        # Store all ensemble results
        all_ensemble_results = []
        
        # Create chunk lookup for document content
        chunk_lookup = {}
        for chunk in chunks:
            doc_id = self._generate_doc_id(chunk, 0)
            chunk_lookup[chunk.get("file_name", "unknown")] = chunk.get("content", "")
        
        # Run vanilla validation multiple times
        for run_idx in range(ensemble_count):
            logger.info(f"[Vanilla Ensemble] Run {run_idx + 1}/{ensemble_count}")
            
            # Group atomic info by chunk for bulk validation
            chunk_validation_tasks = {}
            
            for chunk_id, atomic_list in atomic_info_map.items():
                # Get document content and title for this chunk
                file_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
                doc_content = chunk_lookup.get(file_name, "")
                doc_title = file_name
                
                # Use all atomic info (keyword filtering removed)
                filtered_atomic_list = atomic_list
                
                if filtered_atomic_list:
                    chunk_validation_tasks[chunk_id] = (filtered_atomic_list, doc_content, doc_title, language, model)
            
            # Process bulk validation for this run
            run_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit bulk validation tasks (one per chunk)
                future_to_chunk = {
                    executor.submit(self._validate_bulk_atomic_info_vanilla, atomic_list, doc_content, doc_title, language, model): 
                    chunk_id for chunk_id, (atomic_list, doc_content, doc_title, language, model) in chunk_validation_tasks.items()
                }
                
                # Process results with progress bar
                with tqdm(total=len(chunk_validation_tasks), desc=f"Vanilla Ensemble Run {run_idx + 1}", unit="chunks") as pbar:
                    for future in as_completed(future_to_chunk):
                        chunk_id = future_to_chunk[future]
                        
                        try:
                            validated_atomic_list = future.result()
                            
                            # Collect results for this run
                            for validated_atomic_info in validated_atomic_list:
                                if validated_atomic_info.overall_quality_score > 0.0:
                                    run_results.append((validated_atomic_info, chunk_id, run_idx))
                                    
                        except Exception as e:
                            logger.error(f"[Vanilla Ensemble] Error validating chunk {chunk_id} in run {run_idx}: {e}")
                        
                        # Update progress bar
                        current_cost = self.llm_client.get_cost_summary().total_cost_usd
                        pbar.set_postfix({
                            'Cost': f'${current_cost:.4f}',
                            'Items': len(run_results),
                            'Run': f'{run_idx + 1}/{ensemble_count}'
                        })
                        pbar.update(1)
            
            all_ensemble_results.extend(run_results)
            logger.info(f"[Vanilla Ensemble] Run {run_idx + 1} completed: {len(run_results)} items")
        
        # Aggregate ensemble results - average scores for same atomic info
        ensemble_aggregated = {}  # chunk_id -> {content -> aggregated_atomic_info}
        
        for atomic_info, chunk_id, run_idx in all_ensemble_results:
            if chunk_id not in ensemble_aggregated:
                ensemble_aggregated[chunk_id] = {}
            
            content_key = atomic_info.content
            if content_key not in ensemble_aggregated[chunk_id]:
                # First occurrence - initialize with current scores
                ensemble_aggregated[chunk_id][content_key] = {
                    'atomic_info': AtomicInfo(
                        content=atomic_info.content,
                        chunk_id=chunk_id,
                        overall_quality_score=atomic_info.overall_quality_score
                    ),
                    'scores': [atomic_info],
                    'count': 1
                }
            else:
                # Subsequent occurrences - accumulate scores
                existing = ensemble_aggregated[chunk_id][content_key]
                existing['scores'].append(atomic_info)
                existing['count'] += 1
                
                # Update averaged scores
                total_quality = sum(item.overall_quality_score for item in existing['scores'])
                
                count = existing['count']
                existing['atomic_info'].overall_quality_score = total_quality / count
        
        # Convert to final validated atomic info map
        validated_atomic_info_map = {}
        for chunk_id, content_dict in ensemble_aggregated.items():
            validated_atomic_info_map[chunk_id] = []
            for content_key, aggregated_data in content_dict.items():
                atomic_info = aggregated_data['atomic_info']
                # Apply same threshold filtering as vanilla
                if atomic_info.overall_quality_score >= VALIDITY_CONFIDENCE_THRESHOLD:
                    validated_atomic_info_map[chunk_id].append(atomic_info)
        
        # DEBUG: Final summary
        total_ensemble_items = len(all_ensemble_results)
        total_final_items = sum(len(items) for items in validated_atomic_info_map.values())
        logger.info(f"[Vanilla Ensemble] FINAL SUMMARY:")
        logger.info(f"[Vanilla Ensemble] Total ensemble results: {total_ensemble_items}")
        logger.info(f"[Vanilla Ensemble] Final aggregated items: {total_final_items}")
        logger.info(f"[Vanilla Ensemble] Average runs per item: {total_ensemble_items / total_final_items if total_final_items > 0 else 0:.1f}")
        
        return validated_atomic_info_map

    def _validate_bulk_atomic_info_vanilla(
        self, 
        atomic_info_list: List[AtomicInfo], 
        doc_content: str, 
        doc_title: str,
        language: str, 
        model: str,
        fewshot_examples: Optional[str] = None
    ) -> List[AtomicInfo]:
        """Validate multiple atomic information units using Vanilla approach (single overall quality score)"""
        
        if not atomic_info_list:
            return atomic_info_list
        
        try:
            # Process ALL atomic info
            logger.info(f"[Vanilla] Processing all {len(atomic_info_list)} atomic info items in chunk for vanilla validation")
            
            # Build numbered atomic info list for prompt with title and content
            atomic_info_numbered_list = []
            for i, atomic_info in enumerate(atomic_info_list, 1):
                # Include document title if available
                title_prefix = getattr(atomic_info, 'document_title', '') or doc_title
                if title_prefix:
                    content_with_title = f"{title_prefix}: {atomic_info.content}"
                else:
                    content_with_title = atomic_info.content
                atomic_info_numbered_list.append(f"{i}. \"{content_with_title}\"")
            atomic_info_list_str = "\n".join(atomic_info_numbered_list)
            
            # Generate validation prompt (Vanilla, with optional few-shot)
            prompt = generate_prompt(
                PromptType.VERIFY_ATOMIC_INFO_VANILLA,
                atomic_info_list=atomic_info_list_str,
                atomic_count=len(atomic_info_list),
                doc_title=doc_title,
                few_shot_examples=fewshot_examples or "",
                language=language
            )
            
            # Call LLM for bulk validation with score completeness validation
            response = self.llm_client.call_api_with_score_validation(
                prompt=prompt, 
                validation_type='vanilla', 
                expected_count=len(atomic_info_list),
                model=model,
                max_retries=3
            )
            
            # DEBUG: Log raw response
            logger.info(f"[DEBUG Vanilla] LLM response length: {len(response) if response else 0} chars")
            
            # Parse response (expecting JSON array)
            parsed_response = clean_and_parse_json(response)
            
            # DEBUG: Log parsed response info
            logger.info(f"[DEBUG Vanilla] Parsed response type: {type(parsed_response)}, length: {len(parsed_response) if isinstance(parsed_response, list) else 'N/A'}")
            
            if not isinstance(parsed_response, list):
                logger.error(f"[DEBUG Vanilla] Expected JSON array, got {type(parsed_response)}")
                raise ValueError(f"Expected JSON array, got {type(parsed_response)}")
            
            if len(parsed_response) != len(atomic_info_list):
                logger.warning(f"[Vanilla] Response count mismatch: expected {len(atomic_info_list)}, got {len(parsed_response)}")
                logger.debug(f"[DEBUG Vanilla] This chunk will have missing scores for some items")
            
            # Update atomic_info objects with validation results
            for i, atomic_info in enumerate(atomic_info_list):
                if i < len(parsed_response):
                    result = parsed_response[i]
                    
                    # Extract overall quality score
                    overall_quality_score = result.get("overall_quality_score", 0.0)
                    reasoning = result.get("reasoning", "No reasoning provided")
                    
                    # DEBUG: Log individual score extraction
                    available_keys = list(result.keys()) if isinstance(result, dict) else []
                    logger.debug(f"[DEBUG Vanilla] Item {i+1} available keys: {available_keys}")
                    logger.debug(f"[DEBUG Vanilla] Item {i+1} overall_quality_score: {overall_quality_score}")
                    
                    # Update atomic_info with validation results
                    atomic_info.overall_quality_score = overall_quality_score
                    # Set all individual scores to the overall quality score for consistency
                    atomic_info.validity_score = overall_quality_score
                    atomic_info.completeness_score = overall_quality_score  
                    atomic_info.specificity_score = overall_quality_score
                    atomic_info.clarity_score = overall_quality_score
                    atomic_info.questionability_score = overall_quality_score
                    atomic_info.overall_confidence = overall_quality_score
                    atomic_info.validation_reasoning = reasoning
                else:
                    # No response for this item, set zero scores
                    atomic_info.overall_quality_score = 0.0
                    atomic_info.validity_score = 0.0
                    atomic_info.completeness_score = 0.0
                    atomic_info.specificity_score = 0.0
                    atomic_info.clarity_score = 0.0
                    atomic_info.questionability_score = 0.0
                    atomic_info.overall_confidence = 0.0
                    atomic_info.validation_reasoning = "No LLM response received"
            
            return atomic_info_list
            
        except Exception as e:
            logger.error(f"[Vanilla] Error during bulk validation: {e}")
            # Set all scores to 0 on error
            for atomic_info in atomic_info_list:
                atomic_info.overall_quality_score = 0.0
                atomic_info.validity_score = 0.0
                atomic_info.completeness_score = 0.0
                atomic_info.specificity_score = 0.0
                atomic_info.clarity_score = 0.0
                atomic_info.questionability_score = 0.0
                atomic_info.overall_confidence = 0.0
                atomic_info.validation_reasoning = f"Error during validation: {str(e)}"
            return atomic_info_list

    def _step7_prepare_diverse_pool(
        self,
        redundancy_mapping: Dict[str, RedundancyMapping],
        num_information: int,
    ) -> List[Dict[str, Any]]:
        all_items = []
        for mapping in redundancy_mapping.values():
            redundant_chunk_ids = {mapping.chunk_id}
            if mapping.redundant_items != ["unique"]:
                for redundant_id in mapping.redundant_items:
                    redundant_mapping = redundancy_mapping.get(redundant_id)
                    if redundant_mapping:
                        redundant_chunk_ids.add(redundant_mapping.chunk_id)

            redundancy_count = len(redundant_chunk_ids) - 1
            all_items.append(
                {
                    "atomic_info_id": mapping.atomic_info_id,
                    "content": mapping.content,
                    "chunk_id": mapping.chunk_id,
                    "redundant_items": mapping.redundant_items,
                    "similarity_scores": mapping.similarity_scores,
                    "redundancy_count": redundancy_count,
                }
            )

        redundant_groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        unique_items: List[Dict[str, Any]] = []

        for item in all_items:
            if item["redundant_items"] == ["unique"]:
                unique_items.append(item)
                continue

            group_key = tuple(sorted(item["redundant_items"] + [item["atomic_info_id"]]))
            redundant_groups.setdefault(group_key, []).append(item)

        representatives: List[Dict[str, Any]] = []
        for group_items in redundant_groups.values():
            group_items.sort(key=lambda x: x["redundancy_count"], reverse=True)
            representatives.append(group_items[0])

        diverse_items = unique_items + representatives
        diverse_items.sort(key=lambda x: x["redundancy_count"], reverse=True)

        if len(diverse_items) < num_information:
            logger.warning(
                "[RARE Step 7] Not enough diverse items (%d) for %d information units",
                len(diverse_items),
                num_information,
            )

        return diverse_items

    def _step7_generate_llm_selection_samples(
        self,
        diverse_items: List[Dict[str, Any]],
        redundancy_mapping: Dict[str, RedundancyMapping],
        chunk_data_dict: Dict[str, Dict[str, Any]],
        args: SimpleNamespace,
        actual_num_sample: int,
        tracker: _Step7ValidationTracker,
    ) -> Tuple[List[_Step7QuestionSample], List[_Step7QuestionSample]]:
        if len(diverse_items) < args.input_pool_size:
            logger.warning(
                "[RARE Step 7] Not enough diverse items (%d) for pool size %d",
                len(diverse_items),
                args.input_pool_size,
            )
            return [], []

        preselected_pools = []
        for index in range(actual_num_sample):
            if len(diverse_items) < args.input_pool_size:
                break
            pool_items = self._step7_sample_chunk_diverse_pool(diverse_items, args.input_pool_size)
            preselected_pools.append(
                {
                    "sample_id": f"llm_sample_{index + 1:03d}",
                    "pool_items": pool_items,
                }
            )

        if not preselected_pools:
            logger.warning("[RARE Step 7] No pools could be generated")
            return [], []

        generated_samples: List[_Step7QuestionSample] = []
        extended_samples: List[_Step7QuestionSample] = []

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_pool = {
                executor.submit(
                    self._step7_generate_single_sample,
                    pool_info["sample_id"],
                    pool_info["pool_items"],
                    redundancy_mapping,
                    chunk_data_dict,
                    args,
                    tracker,
                ): pool_info
                for pool_info in preselected_pools
            }

            with tqdm(total=len(preselected_pools), desc="Data Generation Pipeline", unit="samples") as pbar:
                for future in as_completed(future_to_pool):
                    result = future.result()
                    if result:
                        best_sample, validated_samples = result
                        if best_sample:
                            generated_samples.append(best_sample)
                        if validated_samples:
                            extended_samples.extend(validated_samples)

                    with tracker.lock:
                        success_count = len(generated_samples)
                        cost = self.llm_client.get_total_cost_usd()
                        pbar.set_postfix(
                            {
                                "Gen": f"{tracker.progress['generation']}/{len(preselected_pools)}",
                                "Cons": f"{tracker.progress['consistency']}/{len(preselected_pools)}",
                                "Ans": f"{tracker.progress['answerability']}/{len(preselected_pools)}",
                                "Val": f"{tracker.progress['validation']}/{len(preselected_pools)}",
                                "Success": f"{success_count}",
                                "Cost": f"${cost:.4f}",
                            }
                        )
                    pbar.update(1)

        return generated_samples, extended_samples

    def _step7_sample_chunk_diverse_pool(
        self,
        diverse_items: List[Dict[str, Any]],
        pool_size: int,
    ) -> List[Dict[str, Any]]:
        chunk_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in diverse_items:
            chunk_id = item.get("chunk_id", "unknown")
            chunk_groups.setdefault(chunk_id, []).append(item)

        representatives: List[Dict[str, Any]] = []
        for items in chunk_groups.values():
            items.sort(key=lambda x: x.get("redundancy_count", 0), reverse=True)
            representatives.append(items[0])

        if len(representatives) >= pool_size:
            return random.sample(representatives, pool_size)

        pool_items = list(representatives)
        remaining_needed = pool_size - len(pool_items)
        if remaining_needed > 0:
            remaining_items = [item for item in diverse_items if item not in pool_items]
            if remaining_items:
                extra_items = random.sample(remaining_items, min(remaining_needed, len(remaining_items)))
                pool_items.extend(extra_items)

        random.shuffle(pool_items)
        return pool_items

    def _step7_generate_single_sample(
        self,
        sample_id: str,
        pool_items: List[Dict[str, Any]],
        redundancy_mapping: Dict[str, RedundancyMapping],
        chunk_data_dict: Dict[str, Dict[str, Any]],
        args: SimpleNamespace,
        tracker: _Step7ValidationTracker,
    ) -> Optional[Tuple[_Step7QuestionSample, List[_Step7QuestionSample]]]:
        try:
            input_sentences = []
            for index, item in enumerate(pool_items, 1):
                chunk_id = item["chunk_id"]
                title = "_".join(chunk_id.split("_")[:-2]) if "_" in chunk_id else chunk_id
                content = f"{title}: {item['content']}" if title else item["content"]
                input_sentences.append(f"{index}. {content}")

            prompt = generate_prompt(
                PromptType.GENERATE_MULTIHOP_QUESTION_WITH_LLM_SELECTION,
                input_sentences="\n".join(input_sentences),
                input_pool_size=args.input_pool_size,
                num_information=args.num_information,
                num_questions=args.output_questions,
            )
            response = self.llm_client.call_api(prompt, args.generation_model, validate_json=True)
            generation_result = clean_and_parse_json(response)
            tracker.increment_progress("generation")

            if not generation_result or "generated_questions" not in generation_result:
                logger.debug("[RARE Step 7] Sample %s generation failed", sample_id)
                return None

            generated_questions = generation_result.get("generated_questions", [])
            candidates = []
            question_atomic_map: Dict[int, List[Dict[str, Any]]] = {}

            for question_index, question_data in enumerate(generated_questions):
                selected_indices = question_data.get("selected_items", [])
                selected_infos = []
                for idx in selected_indices:
                    if 1 <= idx <= len(pool_items):
                        selected_infos.append(pool_items[idx - 1])

                if len(selected_infos) == args.num_information:
                    question_atomic_map[question_index] = selected_infos
                    candidates.append(
                        {
                            "generated_question": question_data.get("generated_question", ""),
                            "generated_answer": question_data.get("generated_answer", ""),
                            "reasoning": f"LLM Selection: {question_data.get('reasoning', '')}",
                            "selected_items": selected_indices,
                            "question_index": question_index,
                        }
                    )

            if not candidates or not question_atomic_map:
                logger.debug("[RARE Step 7] Sample %s produced no valid candidates", sample_id)
                return None

            gold_sentences = []
            first_question_infos = list(question_atomic_map.values())[0]
            for item in first_question_infos:
                gold_sentences.append(
                    {
                        "atomic_info_id": item["atomic_info_id"],
                        "content": item["content"],
                        "chunk_id": item["chunk_id"],
                        "title": item["chunk_id"],
                        "sentence": item["content"],
                    }
                )

            paraphrased_candidates = []
            for candidate in candidates:
                candidate_copy = candidate.copy()
                candidate_copy['paraphrasing_strategy'] = 'SKIPPED'
                paraphrased_candidates.append(candidate_copy)

            filter_result = self.filter_questions_logical_consistency(
                gold_sentences=gold_sentences,
                questions=paraphrased_candidates,
                model=args.filter_model,
            )

            logical_types = [
                "contextual_independence",
                "answer_exclusion",
                "information_equivalence",
                "question_ambiguity",
            ]

            if isinstance(filter_result, dict) and "valid_questions" in filter_result:
                filtered_candidates = filter_result["valid_questions"]
                reasoning_candidates = filter_result.get("valid_questions_with_reasoning", [])
                filtered_by_type = filter_result.get("filtered_questions_by_type", {})

                for candidate in filtered_candidates:
                    question_text = candidate.get("generated_question", "")
                    for reasoning_candidate in reasoning_candidates:
                        if reasoning_candidate.get("generated_question", "") == question_text:
                            candidate["context_independence_reasoning"] = reasoning_candidate.get(
                                "contextual_independence_reasoning",
                                reasoning_candidate.get("context_independence_reasoning", ""),
                            )
                            candidate["answer_exclusion_reasoning"] = reasoning_candidate.get(
                                "answer_exclusion_reasoning",
                                reasoning_candidate.get("circular_definition_reasoning", ""),
                            )
                            equivalence_reason = reasoning_candidate.get(
                                "information_equivalence_reasoning",
                                reasoning_candidate.get("information_completeness_reasoning", ""),
                            )
                            candidate["information_equivalence_reasoning"] = equivalence_reason
                            candidate["information_completeness_reasoning"] = equivalence_reason
                            candidate["question_ambiguity_reasoning"] = reasoning_candidate.get(
                                "question_ambiguity_reasoning", ""
                            )
                            break

                for question in paraphrased_candidates:
                    question_text = question.get("generated_question", "")
                    for logical_type in logical_types:
                        orchestrator_key = STEP7_LOGICAL_TYPE_MAPPING[logical_type]
                        failed_questions = filtered_by_type.get(orchestrator_key, [])
                        failed = any(
                            item["question"].get("generated_question", "") == question_text for item in failed_questions
                        )
                        if failed:
                            reason = next(
                                (
                                    item["reasoning"]
                                    for item in failed_questions
                                    if item["question"].get("generated_question", "") == question_text
                                ),
                                f"Failed {logical_type.replace('_', ' ')} check",
                            )
                            tracker.record_logical_failure(
                                sample_id,
                                question_text,
                                logical_type,
                                reason,
                                question.get("selected_items", []),
                                question.get("question_id", ""),
                                None,
                                self._step7_get_selected_contents(question.get("selected_items", []), pool_items),
                                question.get("generated_answer", ""),
                            )
                        else:
                            tracker.record_logical_success(logical_type)
            else:
                filtered_candidates = filter_result if filter_result else []
                for question in paraphrased_candidates:
                    for logical_type in logical_types:
                        tracker.record_logical_failure(
                            sample_id,
                            question.get("generated_question", ""),
                            logical_type,
                            "Question filtered out in batch logical consistency check",
                            question.get("selected_items", []),
                            question.get("question_id", ""),
                            None,
                            self._step7_get_selected_contents(question.get("selected_items", []), pool_items),
                            question.get("generated_answer", ""),
                        )

            if filtered_candidates:
                tracker.increment_progress("consistency")
            else:
                logger.debug(
                    "[RARE Step 7] Sample %s failed logical consistency for all %d candidates",
                    sample_id,
                    len(paraphrased_candidates),
                )
                return None

            questions_with_chunks = []
            for candidate in filtered_candidates:
                question_index = candidate.get("question_index")
                chunk_ids = []
                if question_index is not None and question_index in question_atomic_map:
                    for info in question_atomic_map[question_index]:
                        chunk_id = info["chunk_id"]
                        if chunk_id in chunk_data_dict:
                            chunk_ids.append(chunk_id)
                questions_with_chunks.append({"question": candidate.get("generated_question", ""), "chunk_ids": chunk_ids})

            answerability_results = self.check_question_answerability(
                questions_with_chunks=questions_with_chunks,
                chunk_data_dict=chunk_data_dict,
                model=args.answerability_model,
            )

            answerable_candidates = []
            unanswerable_candidates = []
            for index, candidate in enumerate(filtered_candidates):
                if index < len(answerability_results):
                    result = answerability_results[index]
                    if result and result.get("answerability_check", "fail") == "pass":
                        updated = candidate.copy()
                        if "generated_answer" in result:
                            updated["generated_answer"] = result["generated_answer"]
                        updated["answerability_reasoning"] = result.get("reasoning", "")
                        answerable_candidates.append(updated)
                        tracker.record_answerability_success()
                    else:
                        unanswerable_candidates.append(candidate)
                        tracker.record_answerability_failure(
                            sample_id,
                            candidate.get("generated_question", ""),
                            candidate.get("selected_items", []),
                            questions_with_chunks[index].get("chunk_ids", []),
                            result.get("reasoning", "") if result else "No answerability result returned",
                            self._step7_get_selected_contents(candidate.get("selected_items", []), pool_items),
                            candidate.get("generated_answer", ""),
                        )
                else:
                    unanswerable_candidates.append(candidate)
                    tracker.record_answerability_failure(
                        sample_id,
                        candidate.get("generated_question", ""),
                        candidate.get("selected_items", []),
                        [],
                        "No answerability check performed",
                        self._step7_get_selected_contents(candidate.get("selected_items", []), pool_items),
                        candidate.get("generated_answer", ""),
                    )

            if answerable_candidates:
                tracker.increment_progress("answerability")
            else:
                logger.debug("[RARE Step 7] Sample %s has no answerable candidates", sample_id)
                return None

            validation_scores = self.validate_multihop_questions_batch_separate(
                gold_sentences=gold_sentences,
                questions=answerable_candidates,
                model=args.validation_model,
                max_workers=args.max_workers,
            )

            if validation_scores:
                for index, candidate in enumerate(answerable_candidates):
                    if index < len(validation_scores):
                        scores = validation_scores[index]
                        candidate.update(
                            {
                                "connectivity_score": scores.get("connectivity_score", 0.5),
                                "fluency_score": scores.get("fluency_score", 0.5),
                                "essentiality_score": scores.get("essentiality_score", 0.5),
                                "validity_score": scores.get("validity_score", 0.5),
                            }
                        )
            else:
                for candidate in answerable_candidates:
                    candidate.update(
                        {
                            "connectivity_score": 0.8,
                            "fluency_score": 0.8,
                            "essentiality_score": 0.8,
                            "validity_score": 0.8,
                        }
                    )

            if answerable_candidates:
                tracker.increment_progress("validation")

            if len(answerable_candidates) == 1:
                answerable_candidates[0]["rrf_score"] = 4.0
            else:
                ranks = {"connectivity_score": {}, "fluency_score": {}, "essentiality_score": {}, "validity_score": {}}
                for metric in ranks.keys():
                    sorted_candidates = sorted(answerable_candidates, key=lambda x: x.get(metric, 0), reverse=True)
                    ranks[metric] = {id(candidate): position + 1 for position, candidate in enumerate(sorted_candidates)}
                for candidate in answerable_candidates:
                    identifier = id(candidate)
                    candidate["rrf_score"] = sum(1.0 / ranks[metric][identifier] for metric in ranks)

            best_question = max(answerable_candidates, key=lambda x: x.get("rrf_score", 0))

            best_question_index = best_question.get("question_index")
            best_infos = (
                question_atomic_map.get(best_question_index, first_question_infos)
                if best_question_index is not None and best_question_index in question_atomic_map
                else first_question_infos
            )

            gold_chunks = self._step7_calculate_gold_chunks(
                selected_infos=best_infos,
                redundancy_mapping=redundancy_mapping,
            )

            atomic_info_list = [
                _Step7AtomicInfoData(
                    atomic_id=item["atomic_info_id"],
                    content=item["content"],
                    chunk_id=item["chunk_id"],
                )
                for item in best_infos
            ]

            best_sample = _Step7QuestionSample(
                sample_id=sample_id,
                question=best_question.get("generated_question", ""),
                answer=best_question.get("generated_answer", ""),
                connectivity_score=best_question.get("connectivity_score", 0.0),
                fluency_score=best_question.get("fluency_score", 0.0),
                essentiality_score=best_question.get("essentiality_score", 0.0),
                validity_score=best_question.get("validity_score", 0.0),
                rrf_score=best_question.get("rrf_score", 0.0),
                atomic_info_list=atomic_info_list,
                gold_chunks=gold_chunks,
                generation_reasoning=best_question.get("reasoning", ""),
                num_information_used=len(atomic_info_list),
                context_independence_reasoning=best_question.get("context_independence_reasoning", ""),
                circular_definition_reasoning=best_question.get("answer_exclusion_reasoning", ""),
                information_completeness_reasoning=best_question.get("information_completeness_reasoning", ""),
                information_equivalence_reasoning=best_question.get("information_equivalence_reasoning", ""),
                question_ambiguity_reasoning=best_question.get("question_ambiguity_reasoning", ""),
                answerability_reasoning=best_question.get("answerability_reasoning", ""),
            )

            extended_samples = []
            for index, candidate in enumerate(answerable_candidates):
                question_index = candidate.get("question_index")
                selected_infos = (
                    question_atomic_map.get(question_index, [])
                    if question_index is not None and question_index in question_atomic_map
                    else best_infos
                )
                candidate_gold_chunks = self._step7_calculate_gold_chunks(
                    selected_infos=selected_infos,
                    redundancy_mapping=redundancy_mapping,
                )
                candidate_atomic_list = [
                    _Step7AtomicInfoData(
                        atomic_id=item["atomic_info_id"],
                        content=item["content"],
                        chunk_id=item["chunk_id"],
                    )
                    for item in selected_infos
                ]
                extended_samples.append(
                    _Step7QuestionSample(
                        sample_id=f"{sample_id}_candidate_{index + 1}",
                        question=candidate.get("generated_question", ""),
                        answer=candidate.get("generated_answer", ""),
                        connectivity_score=candidate.get("connectivity_score", 0.0),
                        fluency_score=candidate.get("fluency_score", 0.0),
                        essentiality_score=candidate.get("essentiality_score", 0.0),
                        validity_score=candidate.get("validity_score", 0.0),
                        rrf_score=candidate.get("rrf_score", 0.0),
                        atomic_info_list=candidate_atomic_list,
                        gold_chunks=candidate_gold_chunks,
                        generation_reasoning=candidate.get("reasoning", ""),
                        num_information_used=len(candidate_atomic_list),
                        context_independence_reasoning=candidate.get("context_independence_reasoning", ""),
                        circular_definition_reasoning=candidate.get("answer_exclusion_reasoning", ""),
                        information_completeness_reasoning=candidate.get("information_completeness_reasoning", ""),
                        information_equivalence_reasoning=candidate.get("information_equivalence_reasoning", ""),
                        question_ambiguity_reasoning=candidate.get("question_ambiguity_reasoning", ""),
                        answerability_reasoning=candidate.get("answerability_reasoning", ""),
                    )
                )

            return best_sample, extended_samples

        except Exception as exc:
            logger.error("[RARE Step 7] Sample %s pipeline error: %s", sample_id, str(exc)[:100])
            return None

    def _step7_get_selected_contents(
        self,
        selected_items: List[int],
        pool_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        contents = []
        for index in selected_items:
            if 1 <= index <= len(pool_items):
                item = pool_items[index - 1]
                contents.append(
                    {
                        "index": index,
                        "atomic_info_id": item.get("atomic_info_id", ""),
                        "content": item.get("content", ""),
                        "chunk_id": item.get("chunk_id", ""),
                    }
                )
        return contents

    def _step7_calculate_gold_chunks(
        self,
        selected_infos: List[Dict[str, Any]],
        redundancy_mapping: Dict[str, RedundancyMapping],
    ) -> List[List[str]]:
        gold_chunks: List[List[str]] = []
        for item in selected_infos:
            related_chunks = {item["chunk_id"]}
            redundant_items = item.get("redundant_items", [])
            if redundant_items != ["unique"]:
                for redundant_id in redundant_items:
                    redundant_mapping = redundancy_mapping.get(redundant_id)
                    if redundant_mapping:
                        related_chunks.add(redundant_mapping.chunk_id)

            chunk_list = sorted(related_chunks)
            if chunk_list:
                gold_chunks.append(chunk_list)
        return gold_chunks

    def _convert_chunks_to_dict(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        chunk_data: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            file_name = chunk.get("file_name", "")
            page_no = chunk.get("page_no", 0)
            sub_text_index = str(chunk.get("sub_text_index", "1"))
            if not file_name or not page_no:
                continue
            base_name = file_name.replace(".pdf", "")
            chunk_id = f"{base_name}_page{page_no:03d}_chunk{sub_text_index.zfill(3)}"
            chunk_data[chunk_id] = {
                "source_title": base_name,
                "content": chunk.get("content", ""),
                "page_number": page_no,
            }
        return chunk_data

    def _load_chunk_data_file(self, chunk_file: Path) -> Dict[str, Dict[str, Any]]:
        if not chunk_file.exists():
            logger.warning("[RARE Step 7] Chunk file not found: %s", chunk_file)
            return {}
        with open(chunk_file, "r", encoding="utf-8") as fp:
            chunks = json.load(fp)
        if isinstance(chunks, dict):
            chunks = list(chunks.values())
        if not isinstance(chunks, list):
            logger.warning("[RARE Step 7] Unexpected chunk file format: %s", chunk_file)
            return {}
        return self._convert_chunks_to_dict(chunks)