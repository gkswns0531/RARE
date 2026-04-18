import logging
import threading

from rare_const import EXCHANGE_RATE, MODEL_PRICING
from rare_entities import CostSummary

logger = logging.getLogger(__name__)


class CostTracker:
    """Track cost and usage for LLM API calls in RARE (Thread-Safe)"""
    
    def __init__(self):
        self.summary = CostSummary()
        self._lock = threading.Lock()  # Thread safety for multi-threading

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate cost in USD for given tokens and model"""
        if model not in MODEL_PRICING:
            logger.warning(f"Unknown model: {model}, using default pricing")
            model = "gpt5_nano"  # fallback to cheapest
        
        pricing = MODEL_PRICING[model]
        prompt_cost = (prompt_tokens / 1000) * pricing["input"]
        completion_cost = (completion_tokens / 1000) * pricing["output"]
        
        return prompt_cost + completion_cost

    def log_current_cost(self, prompt_tokens: int, completion_tokens: int, model: str, cost_usd: float):
        """Log current API call cost"""
        # logger.info(f"API Call - Model: {model}, Tokens: {prompt_tokens + completion_tokens} "
        #            f"(P:{prompt_tokens}, C:{completion_tokens}), Cost: ${cost_usd:.6f}")

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost summary with new API call (Thread-Safe)"""
        cost_usd = self.calculate_cost(prompt_tokens, completion_tokens, model)
        
        with self._lock:  # Atomic update to prevent race conditions
            self.summary.total_calls += 1
            self.summary.prompt_tokens += prompt_tokens
            self.summary.completion_tokens += completion_tokens
            self.summary.total_tokens += prompt_tokens + completion_tokens
            self.summary.total_cost_usd += cost_usd
            self.summary.total_cost_krw += cost_usd * EXCHANGE_RATE
            
            # Track per-model costs
            if model in self.summary.model_types:
                self.summary.model_types[model] += cost_usd
            else:
                self.summary.model_types[model] = cost_usd

    def get_current_cost(self) -> CostSummary:
        """Get current cost summary"""
        return self.summary

    def reset_cost(self):
        """Reset cost tracking (Thread-Safe)"""
        with self._lock:  # Atomic reset
            self.summary = CostSummary()
