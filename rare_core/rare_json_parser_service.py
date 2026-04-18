import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _try_parse_code_block(llm_response: str) -> Any:
    """Attempt 1: Extract JSON from code blocks with backticks"""
    try:
        # Extract from ```json``` code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_response, re.DOTALL)
        if json_match:
            clean_text = json_match.group(1).strip()
        else:
            clean_text = llm_response.strip()

        # Convert Python booleans to JSON booleans
        clean_text = re.sub(r"\bTrue\b", "true", clean_text)
        clean_text = re.sub(r"\bFalse\b", "false", clean_text)
        clean_text = re.sub(r"\bNone\b", "null", clean_text)

        # Remove surrounding quotes if entire string is quoted
        if clean_text.startswith('"') and clean_text.endswith('"'):
            clean_text = clean_text[1:-1]

        # Handle escaped strings
        clean_text = clean_text.replace('\\"', '"')

        # Remove trailing commas
        clean_text = re.sub(r",(\s*[}\]])", r"\1", clean_text)

        return json.loads(clean_text)
    except Exception:
        return {}


def _try_parse_json_regex(llm_response: str) -> Any:
    """Attempt 2: Extract JSON objects using regex"""
    try:
        pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        matches = re.findall(pattern, llm_response)
        if matches:
            # Select the longest match (usually more complete JSON)
            longest_match = max(matches, key=len)

            # Convert Python booleans to JSON booleans
            longest_match = re.sub(r"\bTrue\b", "true", longest_match)
            longest_match = re.sub(r"\bFalse\b", "false", longest_match)
            longest_match = re.sub(r"\bNone\b", "null", longest_match)

            return json.loads(longest_match)
    except Exception:
        return {}


def _try_parse_v2_specific_fields(llm_response: str) -> Any:
    """Attempt 3: Extract RARE specific fields"""
    try:
        result = {}

        # RARE specific field patterns
        patterns = {
            "question": r'"question"\s*:\s*"([^"]+)"',
            "answer": r'"answer"\s*:\s*"([^"]+)"',
            "atomic_information": r'"atomic_information"\s*:\s*(\[[^\]]+\])',
            "redundant": r'"redundant"\s*:\s*(true|false|True|False)',
            "content": r'"content"\s*:\s*"([^"]+)"',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                value = match.group(1)

                # Handle booleans
                if key == "redundant" and value.lower() in ("true", "false"):
                    result[key] = value.lower() == "true"
                # Handle arrays
                elif key == "atomic_information" and value.startswith("[") and value.endswith("]"):
                    try:
                        parsed_array = json.loads(value)
                        result[key] = parsed_array
                    except json.JSONDecodeError:
                        result[key] = value
                # Handle numbers
                    try:
                        result[key] = int(value)
                    except ValueError:
                        result[key] = value
                else:
                    result[key] = value

        return result if result else {}

    except Exception:
        return {}


def clean_and_parse_json(llm_response: str) -> Any:
    """Clean and parse JSON from LLM response for RARE - supports both dict and array"""

    # Check for empty text
    if not llm_response or not llm_response.strip():
        return {}

    # Try to parse as direct JSON first (handles both dict and array)
    try:
        clean_text = llm_response.strip()
        
        # Extract from ```json``` code blocks if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(1).strip()
        
        # Convert Python booleans to JSON booleans
        clean_text = re.sub(r"\bTrue\b", "true", clean_text)
        clean_text = re.sub(r"\bFalse\b", "false", clean_text)
        clean_text = re.sub(r"\bNone\b", "null", clean_text)
        
        # Remove trailing commas
        clean_text = re.sub(r",(\s*[}\]])", r"\1", clean_text)
        
        # Try to parse as JSON (dict or array)
        result = json.loads(clean_text)
        
        # General fix: Convert dict with multiple dict values to list
        if isinstance(result, dict) and len(result) > 1 and all(isinstance(v, dict) for v in result.values()):
            result = list(result.values())
            
        return result
        
    except json.JSONDecodeError:
        pass

    # Attempt 1: Parse code blocks with backticks (backward compatibility)
    result = _try_parse_code_block(llm_response)
    if result:
        return result

    # Attempt 2: Extract JSON objects using regex (backward compatibility)
    result = _try_parse_json_regex(llm_response)
    if result:
        return result

    # Attempt 3: Extract RARE specific fields (backward compatibility)
    result = _try_parse_v2_specific_fields(llm_response)
    if result:
        return result

    return {}
