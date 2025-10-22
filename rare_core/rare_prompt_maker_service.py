from rare_const import PromptType
from rare_core.rare_prompts_service import (
    # MIRA atomic info prompts - USED by RARE
    EXTRACT_ATOMIC_INFO_PROMPT,

    # MIRA atomic info prompts - USED by RARE
    VERIFY_ATOMIC_INFO_VANILLA_PROMPT,

    VERIFY_ATOMIC_INFO_VALIDITY_PROMPT,

    VERIFY_ATOMIC_INFO_VALIDITY_SEPARATE_PROMPT,
    VERIFY_ATOMIC_INFO_COMPLETENESS_SEPARATE_PROMPT,
    VERIFY_ATOMIC_INFO_SPECIFICITY_SEPARATE_PROMPT,
    VERIFY_ATOMIC_INFO_CLARITY_SEPARATE_PROMPT,
    VERIFY_ATOMIC_INFO_QUESTIONABILITY_SEPARATE_PROMPT,

    # Redundancy detection
    DETECT_SEMANTIC_REDUNDANCY_PROMPT,
    
    # LLM-based information selection and multi-hop question generation
    GENERATE_MULTIHOP_QUESTION_WITH_LLM_SELECTION_PROMPT,
    
    # Question paraphrasing and answerability
    PARAPHRASE_QUESTIONS_PROMPT,
    FILTER_ANSWERABILITY_PROMPT,
    
    # Query decomposition
    DECOMPOSE_QUERY_PROMPT,
    ESTIMATE_DECOMPOSITION_COUNT_PROMPT,
    
    # Multi-hop question separate validation (4 separate calls - answerability removed)
    VALIDATE_MULTIHOP_QUESTIONS_CONNECTIVITY_SEPARATE_PROMPT,
    VALIDATE_MULTIHOP_QUESTIONS_FLUENCY_SEPARATE_PROMPT,
    VALIDATE_MULTIHOP_QUESTIONS_ESSENTIALITY_SEPARATE_PROMPT,
    VALIDATE_MULTIHOP_QUESTIONS_VALIDITY_SEPARATE_PROMPT,
    
    # Logical consistency filtering for questions (4 separate calls)
    FILTER_CONTEXTUAL_INDEPENDENCE_PROMPT,
    FILTER_ANSWER_EXCLUSION_PROMPT,
    FILTER_INFORMATION_EQUIVALENCE_PROMPT,
    FILTER_QUESTION_AMBIGUITY_PROMPT,
    
    # Information completeness filtering for atomic info (single call)
    FILTER_ATOMIC_INFORMATION_COMPLETENESS_PROMPT,

)


def generate_prompt(prompt_type: PromptType, **kwargs) -> str:
    """Generate prompt based on type and parameters for RARE"""
    language = kwargs.get("language", "English")

    match prompt_type:
        
        case PromptType.EXTRACT_ATOMIC_INFO:
            prompt = EXTRACT_ATOMIC_INFO_PROMPT.format(
                doc_title=kwargs.get("doc_title", ""),
                doc_content=kwargs["doc_content"],
                language=language,
            )
        
        case PromptType.VERIFY_ATOMIC_INFO_VANILLA:
            prompt = VERIFY_ATOMIC_INFO_VANILLA_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                atomic_count=kwargs.get("atomic_count", "N"),
                doc_title=kwargs.get("doc_title", ""),
                few_shot_examples=kwargs.get("few_shot_examples", ""),
                language=language,
            )

        case PromptType.VERIFY_ATOMIC_INFO_VALIDITY:
            prompt = VERIFY_ATOMIC_INFO_VALIDITY_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                atomic_count=kwargs.get("atomic_count", "N"),
                doc_title=kwargs.get("doc_title", ""),
                doc_content=kwargs["doc_content"],
                few_shot_examples=kwargs.get("few_shot_examples", ""),
                language=language,
            )

        case PromptType.VERIFY_ATOMIC_INFO_RANKING:
            raise ValueError("VERIFY_ATOMIC_INFO_RANKING prompt template not defined in this build")

        case PromptType.VERIFY_ATOMIC_INFO_VALIDITY_SEPARATE:
            prompt = VERIFY_ATOMIC_INFO_VALIDITY_SEPARATE_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                atomic_count=kwargs.get("atomic_count", "N"),
                doc_title=kwargs.get("doc_title", ""),
                doc_content=kwargs["doc_content"],
                few_shot_examples=kwargs.get("few_shot_examples", ""),
                language=language,
            )

        case PromptType.VERIFY_ATOMIC_INFO_COMPLETENESS_SEPARATE:
            prompt = VERIFY_ATOMIC_INFO_COMPLETENESS_SEPARATE_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                atomic_count=kwargs.get("atomic_count", "N"),
                doc_title=kwargs.get("doc_title", ""),
                doc_content=kwargs["doc_content"],
                few_shot_examples=kwargs.get("few_shot_examples", ""),
                language=language,
            )

        case PromptType.VERIFY_ATOMIC_INFO_SPECIFICITY_SEPARATE:
            prompt = VERIFY_ATOMIC_INFO_SPECIFICITY_SEPARATE_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                atomic_count=kwargs.get("atomic_count", "N"),
                doc_title=kwargs.get("doc_title", ""),
                doc_content=kwargs["doc_content"],
                few_shot_examples=kwargs.get("few_shot_examples", ""),
                language=language,
            )

        case PromptType.VERIFY_ATOMIC_INFO_CLARITY_SEPARATE:
            prompt = VERIFY_ATOMIC_INFO_CLARITY_SEPARATE_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                atomic_count=kwargs.get("atomic_count", "N"),
                doc_title=kwargs.get("doc_title", ""),
                doc_content=kwargs["doc_content"],
                few_shot_examples=kwargs.get("few_shot_examples", ""),
                language=language,
            )

        case PromptType.VERIFY_ATOMIC_INFO_QUESTIONABILITY_SEPARATE:
            prompt = VERIFY_ATOMIC_INFO_QUESTIONABILITY_SEPARATE_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                atomic_count=kwargs.get("atomic_count", "N"),
                doc_title=kwargs.get("doc_title", ""),
                doc_content=kwargs["doc_content"],
                few_shot_examples=kwargs.get("few_shot_examples", ""),
                language=language,
            )

        case PromptType.GENERATE_MULTIHOP_QUESTION_WITH_LLM_SELECTION:
            prompt = GENERATE_MULTIHOP_QUESTION_WITH_LLM_SELECTION_PROMPT.format(
                input_pool_size=kwargs.get("input_pool_size", 5),
                num_information=kwargs.get("num_information", 2),
                num_questions=kwargs.get("num_questions", 3),
                input_sentences=kwargs["input_sentences"]
            )

        # Question paraphrasing and answerability
        case PromptType.PARAPHRASE_QUESTIONS:
            prompt = PARAPHRASE_QUESTIONS_PROMPT.format(
                questions_and_answers=kwargs["questions_and_answers"],
                document_titles=kwargs["document_titles"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.VERIFY_QUESTION_ANSWERABILITY:
            prompt = FILTER_ANSWERABILITY_PROMPT.format(
                questions_list=kwargs["questions_list"],
                chunks_with_titles=kwargs["chunks_with_titles"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.DECOMPOSE_QUERY:
            num_decomposed = kwargs.get("num_decomposed", "N")
            if num_decomposed == "N" or num_decomposed is None:
                num_decomposed_str = "an optimal number of"
                exact_count_instruction = "Determine the optimal number of sub-queries needed based on query complexity."
            else:
                num_decomposed_str = str(num_decomposed)
                exact_count_instruction = f"Generate exactly {num_decomposed} decomposed sub-queries that satisfy all quality requirements."
            
            prompt = DECOMPOSE_QUERY_PROMPT.format(
                original_query=kwargs["original_query"],
                num_decomposed=num_decomposed_str,
                exact_count_instruction=exact_count_instruction
            )

        case PromptType.ESTIMATE_DECOMPOSITION_COUNT:
            prompt = ESTIMATE_DECOMPOSITION_COUNT_PROMPT.format(
                original_query=kwargs["original_query"]
            )

        # Multi-hop question separate validation (4 separate calls - answerability removed)
        case PromptType.VALIDATE_MULTIHOP_QUESTIONS_CONNECTIVITY_SEPARATE:
            prompt = VALIDATE_MULTIHOP_QUESTIONS_CONNECTIVITY_SEPARATE_PROMPT.format(
                input_sentences=kwargs["input_sentences"],
                questions_list=kwargs["questions_list"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.VALIDATE_MULTIHOP_QUESTIONS_FLUENCY_SEPARATE:
            prompt = VALIDATE_MULTIHOP_QUESTIONS_FLUENCY_SEPARATE_PROMPT.format(
                input_sentences=kwargs["input_sentences"],
                questions_list=kwargs["questions_list"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.VALIDATE_MULTIHOP_QUESTIONS_ESSENTIALITY_SEPARATE:
            prompt = VALIDATE_MULTIHOP_QUESTIONS_ESSENTIALITY_SEPARATE_PROMPT.format(
                input_sentences=kwargs["input_sentences"],
                questions_list=kwargs["questions_list"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.VALIDATE_MULTIHOP_QUESTIONS_VALIDITY_SEPARATE:
            prompt = VALIDATE_MULTIHOP_QUESTIONS_VALIDITY_SEPARATE_PROMPT.format(
                input_sentences=kwargs["input_sentences"],
                questions_list=kwargs["questions_list"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.FILTER_CONTEXT_ASSUMPTION:
            prompt = FILTER_CONTEXTUAL_INDEPENDENCE_PROMPT.format(
                questions_list=kwargs["questions_list"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.FILTER_CIRCULAR_DEFINITION:
            prompt = FILTER_ANSWER_EXCLUSION_PROMPT.format(
                questions_list=kwargs["questions_list"],
                input_sentences=kwargs["input_sentences"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.FILTER_INFORMATION_COMPLETENESS:
            prompt = FILTER_INFORMATION_EQUIVALENCE_PROMPT.format(
                questions_list=kwargs["questions_list"],
                input_sentences=kwargs["input_sentences"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        case PromptType.FILTER_QUESTION_AMBIGUITY:
            prompt = FILTER_QUESTION_AMBIGUITY_PROMPT.format(
                questions_list=kwargs["questions_list"],
                num_questions=kwargs.get("num_questions", "ALL")
            )

        # Atomic info completeness filtering (single call)
        case PromptType.FILTER_ATOMIC_INFORMATION_COMPLETENESS:
            prompt = FILTER_ATOMIC_INFORMATION_COMPLETENESS_PROMPT.format(
                atomic_info_list=kwargs["atomic_info_list"],
                num_atomic=kwargs.get("num_atomic", "ALL")
            )

        case PromptType.DETECT_SEMANTIC_REDUNDANCY:
            prompt = DETECT_SEMANTIC_REDUNDANCY_PROMPT.format(
                target_info=kwargs["target_info"],
                comparison_info_list=kwargs["comparison_info_list"],
                num_comparisons=kwargs.get("num_comparisons", len(kwargs.get("comparison_info_list", "").split("\n")))
            )

        case _:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompt
