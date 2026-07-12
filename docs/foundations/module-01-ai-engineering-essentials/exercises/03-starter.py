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
    # TODO: Get the encoding for the model
    # encoding = tiktoken.encoding_for_model(model)
    
    # TODO: Encode the text and count tokens
    # tokens = encoding.encode(text)
    # return len(tokens)
    
    pass  # Remove when implemented


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
    # Pricing per 1M tokens
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    
    # TODO: Calculate costs
    # model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    # input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    # output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    # TODO: Return the breakdown
    # return {
    #     "input_cost": input_cost,
    #     "output_cost": output_cost,
    #     "total_cost": input_cost + output_cost,
    #     "input_tokens": input_tokens,
    #     "output_tokens": output_tokens
    # }
    
    pass  # Remove when implemented


# Test your functions!
if __name__ == "__main__":
    text = "Hello! I'm learning about AI tokens and cost optimization."
    
    # Count tokens
    token_count = count_tokens(text)
    print(f"Text: {text}")
    print(f"Tokens: {token_count}")
    
    # Calculate cost for a conversation
    print("\n--- Cost Calculation ---")
    cost = calculate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o-mini")
    print(f"Input: {cost['input_tokens']} tokens")
    print(f"Output: {cost['output_tokens']} tokens")
    print(f"Input cost: ${cost['input_cost']:.6f}")
    print(f"Output cost: ${cost['output_cost']:.6f}")
    print(f"Total cost: ${cost['total_cost']:.6f}")

