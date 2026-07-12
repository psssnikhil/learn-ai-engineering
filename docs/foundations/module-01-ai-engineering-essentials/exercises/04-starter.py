# Prompt Engineering Fundamentals - Exercise
# Practice building well-structured prompts using the ROLE + TASK + CONTEXT + FORMAT + CONSTRAINTS pattern

def build_prompt(role, task, context="", format_instruction="", constraints=""):
    """
    Build a structured prompt from components.

    TODO: Combine the components into a well-formatted prompt string.
    Each non-empty component should be on its own line with a label.

    Example output:
    "Role: You are a Python expert.\nTask: Review this code.\nContext: The code is for a web app."
    """
    # TODO: Build the prompt string
    pass


def analyze_prompt_quality(prompt):
    """
    Analyze a prompt and return a quality score (0-100) based on:
    - Has a clear role (20 points)
    - Has a specific task (20 points)
    - Has context (20 points)
    - Has format instructions (20 points)
    - Has constraints (20 points)

    TODO: Check for keywords indicating each component and calculate score.
    Hints:
    - Role keywords: "you are", "act as", "role"
    - Task keywords: "explain", "write", "create", "analyze", "review", "build"
    - Context keywords: "context", "background", "given that", "for a"
    - Format keywords: "format", "output", "return", "provide as", "list", "table"
    - Constraint keywords: "limit", "must", "should not", "under", "maximum", "only"
    """
    # TODO: Implement prompt quality analysis
    pass


def apply_few_shot(task_description, examples, new_input):
    """
    Create a few-shot prompt by combining examples with a new input.

    Args:
        task_description: What the AI should do
        examples: List of (input, output) tuples
        new_input: The new input to process

    TODO: Build a few-shot prompt that includes:
    1. The task description
    2. Each example formatted as "Input: {input}\nOutput: {output}"
    3. The new input formatted as "Input: {new_input}\nOutput:"

    Return the complete prompt string.
    """
    # TODO: Implement few-shot prompt builder
    pass


# Test your implementations
if __name__ == "__main__":
    # Test 1: Build a prompt
    prompt = build_prompt(
        role="You are a senior Python developer",
        task="Review this code for bugs",
        context="The code is part of a web application",
        format_instruction="List each bug with its line number",
        constraints="Focus only on critical bugs"
    )
    print("=== Built Prompt ===")
    print(prompt)
    print()

    # Test 2: Analyze prompt quality
    good_prompt = "You are an expert data scientist. Analyze this dataset and provide insights. The data is from an e-commerce platform. Format as a bullet list. Limit to top 5 insights."
    bad_prompt = "Tell me about data."

    print("=== Prompt Quality Scores ===")
    print(f"Good prompt score: {analyze_prompt_quality(good_prompt)}")
    print(f"Bad prompt score: {analyze_prompt_quality(bad_prompt)}")
    print()

    # Test 3: Few-shot prompt
    examples = [
        ("Happy", "Positive"),
        ("Terrible", "Negative"),
        ("Okay", "Neutral")
    ]
    few_shot = apply_few_shot(
        "Classify the sentiment of the text.",
        examples,
        "Amazing"
    )
    print("=== Few-Shot Prompt ===")
    print(few_shot)
