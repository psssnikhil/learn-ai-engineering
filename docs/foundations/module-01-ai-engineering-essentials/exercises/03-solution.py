import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model to use for encoding (default: gpt-4o)
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Calculate the cost of an API call.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: The model being used
        
    Returns:
        Dictionary with cost breakdown
    """
    # Pricing per 1M tokens (as of December 2024)
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model
    }


def optimize_prompt(prompt: str, max_tokens: int = 1000) -> str:
    """
    Optimize a prompt by truncating if too long.
    
    Args:
        prompt: The original prompt
        max_tokens: Maximum allowed tokens
        
    Returns:
        Optimized prompt
    """
    token_count = count_tokens(prompt)
    
    if token_count <= max_tokens:
        return prompt
    
    # Truncate to fit within limit
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(prompt)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


class CostTracker:
    """Track costs across multiple API calls"""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
    
    def add_call(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini"):
        """Record a new API call"""
        cost_info = calculate_cost(input_tokens, output_tokens, model)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost_info["total_cost"]
        self.calls += 1
    
    def get_summary(self) -> dict:
        """Get cost summary"""
        return {
            "total_calls": self.calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_call": self.total_cost / self.calls if self.calls > 0 else 0
        }
    
    def print_summary(self):
        """Print cost summary"""
        summary = self.get_summary()
        print("\n=== Cost Summary ===")
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"  Input: {summary['total_input_tokens']:,}")
        print(f"  Output: {summary['total_output_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost']:.6f}")
        print(f"Avg Cost/Call: ${summary['avg_cost_per_call']:.6f}")


# Test your functions!
if __name__ == "__main__":
    print("=== Token Counter ===")
    text = "Hello! I'm learning about AI tokens and cost optimization."
    token_count = count_tokens(text)
    print(f"Text: {text}")
    print(f"Tokens: {token_count}")
    
    print("\n=== Cost Calculation ===")
    cost = calculate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o-mini")
    print(f"Model: {cost['model']}")
    print(f"Input: {cost['input_tokens']:,} tokens → ${cost['input_cost']:.6f}")
    print(f"Output: {cost['output_tokens']:,} tokens → ${cost['output_cost']:.6f}")
    print(f"Total cost: ${cost['total_cost']:.6f}")
    
    print("\n=== Model Comparison ===")
    models = ["gpt-4o", "gpt-4o-mini"]
    for model in models:
        cost = calculate_cost(10000, 5000, model)
        print(f"{model}: ${cost['total_cost']:.4f}")
    
    print("\n=== Cost Tracker Demo ===")
    tracker = CostTracker()
    
    # Simulate multiple API calls
    tracker.add_call(1000, 500, "gpt-4o-mini")
    tracker.add_call(2000, 1000, "gpt-4o-mini")
    tracker.add_call(500, 250, "gpt-4o-mini")
    
    tracker.print_summary()
    
    print("\n=== Prompt Optimization ===")
    long_prompt = "AI " * 1000  # Very long prompt
    print(f"Original tokens: {count_tokens(long_prompt)}")
    
    optimized = optimize_prompt(long_prompt, max_tokens=100)
    print(f"Optimized tokens: {count_tokens(optimized)}")

