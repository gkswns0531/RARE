import openai
from rare_const import DEFAULT_MODEL
from rare_core.rare_cost_tracker_service import CostTracker


class LLMClient:
    """LLM client for RARE with cost tracking"""
    
    def __init__(self, cost_tracker: CostTracker = None, api_key: str = None):
        self.cost_tracker = cost_tracker or CostTracker()
        self.client = openai.OpenAI(api_key=api_key)
        
    def _map_model_name(self, model: str) -> str:
        """Map internal model names to OpenAI model names"""
        model_mapping = {
            "gpt41": "gpt-4.1",
            "gpt41_mini": "gpt-4.1-mini",
            "gpt41_nano": "gpt-4.1-nano",
            
            "gpt5": "gpt-5",
            "gpt5_mini": "gpt-5-mini", 
            "gpt5_nano": "gpt-5-nano",
            
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini",
            "gpt-4.1-nano": "gpt-4.1-nano", 
            "gpt-5": "gpt-5",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-nano": "gpt-5-nano"
        }
        return model_mapping.get(model, model)
    
    def call_api(self, prompt: str, model: str = DEFAULT_MODEL, max_retries: int = 3, validate_json: bool = True) -> str:
        """Make API call to LLM with response validation and retry logic"""
        import time
        import logging
        from .rare_json_parser_service import clean_and_parse_json
        
        openai_model = self._map_model_name(model)
        
        # Build API parameters
        api_params = {
            "model": openai_model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                
                output_text = response.choices[0].message.content
                
                # Extract usage information and track cost IMMEDIATELY after API call
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                cost_usd = self.cost_tracker.calculate_cost(prompt_tokens, completion_tokens, model)
                self.cost_tracker.log_current_cost(prompt_tokens, completion_tokens, model, cost_usd)
                self.cost_tracker.update_cost(prompt_tokens, completion_tokens, model)
                
                # Basic validation
                if not output_text or output_text.strip() == "":
                    raise ValueError("Empty response from API")
                
                # JSON parsing validation (optional) - AFTER cost tracking
                if validate_json:
                    parsed_result = clean_and_parse_json(output_text)
                    if not parsed_result:
                        raise ValueError(f"JSON parsing failed for response: {output_text[:100]}...")
                    
                    # Additional format validation for RARE responses
                    if not self._is_valid_mira_response(parsed_result):
                        raise ValueError(f"Invalid MIRA response format: {type(parsed_result)} - {str(parsed_result)[:100]}...")

                return output_text
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"API call failed after {max_retries} attempts: {e}")
        
        raise Exception(f"LLM API call failed: {str(last_error)}")
    
    def _is_valid_mira_response(self, parsed_result) -> bool:
        """Validate if parsed result is in expected RARE format"""
        try:
            # Accept lists (common for validation results)
            if isinstance(parsed_result, list):
                return True
            
            # Accept dicts with meaningful content
            if isinstance(parsed_result, dict):
                # Empty dict is invalid
                if not parsed_result:
                    return False
                    
                # Dict with common MIRA keys is valid
                valid_keys = {'ranking', 'validation', 'atomic_info', 'question', 'answer', 
                             'validity_score', 'completeness_score', 'specificity_score', 
                             'clarity_score', 'questionability_score', "answerability_score",
                             'effectiveness_score', 'connectivity_score', 'fluency_score', 'essentiality_score'}
                if any(key in parsed_result for key in valid_keys):
                    return True
                
                # Dict with at least some meaningful content
                if len(parsed_result) > 0:
                    return True
                    
            # Other types (string, number, etc.) are invalid
            return False
            
        except Exception:
            return False
    
    def validate_score_completeness(self, parsed_result, validation_type: str, expected_count: int = None) -> tuple[bool, str]:
        """Validate if all required scores are present in the response
        
        Args:
            parsed_result: Parsed JSON response from LLM
            validation_type: 'combined', 'separate_validity', 'separate_completeness', etc.
            expected_count: Expected number of items in the response
            
        Returns:
            (is_valid, error_message): Tuple of validation result and error description
        """
        try:
            if not isinstance(parsed_result, list):
                return False, f"Expected list, got {type(parsed_result)}"
            
            if expected_count is not None and len(parsed_result) != expected_count:
                return False, f"Expected {expected_count} items, got {len(parsed_result)}"
            
            # Define required scores per validation type
            required_scores = {
                'vanilla': ['overall_quality_score'],
                'combined': ['validity_score', 'completeness_score', 'specificity_score', 'clarity_score', 'questionability_score'],
                'separate_validity': ['validity_score'],
                'separate_completeness': ['completeness_score'], 
                'separate_specificity': ['specificity_score'],
                'separate_clarity': ['clarity_score'],
                'separate_questionability': ['questionability_score'],
                # Multi-hop question validation types (hotpot_questions pipeline)
                'separate_connectivity': ['connectivity_score'],
                'separate_fluency': ['fluency_score'],
                'separate_essentiality': ['essentiality_score'],
                # Logical filtering validation types - no specific score requirements, just format validation
                'filter_context_assumption': [],
                'filter_conversion_fallacy': [],
                'filter_information_completeness': [],
                'filter_question_ambiguity': []
            }
            
            required = required_scores.get(validation_type, [])
            if not required:
                return True, "No specific score requirements"
            
            # Check each item in the response
            for i, item in enumerate(parsed_result):
                if not isinstance(item, dict):
                    return False, f"Item {i+1}: expected dict, got {type(item)}"
                
                missing_scores = []
                for score_field in required:
                    if score_field not in item:
                        missing_scores.append(score_field)
                    elif item[score_field] is None:
                        missing_scores.append(f"{score_field} (null)")
                    elif not isinstance(item[score_field], (int, float)):
                        missing_scores.append(f"{score_field} (type: {type(item[score_field])})")
                
                if missing_scores:
                    return False, f"Item {i+1}: missing/invalid scores: {missing_scores}"
            
            return True, "All scores present and valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def call_api_with_score_validation(self, prompt: str, validation_type: str, expected_count: int = None, 
                                       model: str = DEFAULT_MODEL, max_retries: int = 3) -> str:
        """Make API call with score completeness validation and retry logic"""
        import logging
        from .rare_json_parser_service import clean_and_parse_json
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Make regular API call
                response = self.call_api(prompt, model, max_retries=1, validate_json=True)
                
                # Additional score completeness validation
                parsed_result = clean_and_parse_json(response)
                is_valid, error_msg = self.validate_score_completeness(parsed_result, validation_type, expected_count)
                
                if not is_valid:
                    raise ValueError(f"Score validation failed: {error_msg}")
                
                return response
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    logging.warning(f"API call with score validation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    import time
                    time.sleep(wait_time)
                else:
                    logging.error(f"API call with score validation failed after {max_retries} attempts: {e}")
        
        raise Exception(f"LLM API call with score validation failed: {str(last_error)}")
    
    def get_total_cost_usd(self) -> float:
        """Get total cost in USD"""
        return self.cost_tracker.summary.total_cost_usd
    
    def get_total_api_calls(self) -> int:
        """Get total number of API calls made"""
        return self.cost_tracker.summary.total_calls
    
    def get_cost_summary(self):
        """Get detailed cost summary"""
        return self.cost_tracker.get_current_cost()
