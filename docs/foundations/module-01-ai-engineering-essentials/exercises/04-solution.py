# Prompt Engineering Fundamentals - Solution

def build_prompt(role, task, context="", format_instruction="", constraints=""):
    """Build a structured prompt from components."""
    parts = []
    if role:
        parts.append(f"Role: {role}")
    if task:
        parts.append(f"Task: {task}")
    if context:
        parts.append(f"Context: {context}")
    if format_instruction:
        parts.append(f"Format: {format_instruction}")
    if constraints:
        parts.append(f"Constraints: {constraints}")
    return "\n".join(parts)


def analyze_prompt_quality(prompt):
    """Analyze a prompt and return a quality score (0-100)."""
    prompt_lower = prompt.lower()
    score = 0

    # Check for role (20 points)
    role_keywords = ["you are", "act as", "role", "expert", "specialist", "professional"]
    if any(kw in prompt_lower for kw in role_keywords):
        score += 20

    # Check for task (20 points)
    task_keywords = ["explain", "write", "create", "analyze", "review", "build", "describe", "summarize", "generate", "provide"]
    if any(kw in prompt_lower for kw in task_keywords):
        score += 20

    # Check for context (20 points)
    context_keywords = ["context", "background", "given that", "for a", "the data", "this is", "from a", "part of"]
    if any(kw in prompt_lower for kw in context_keywords):
        score += 20

    # Check for format (20 points)
    format_keywords = ["format", "output", "return", "provide as", "list", "table", "bullet", "json", "markdown"]
    if any(kw in prompt_lower for kw in format_keywords):
        score += 20

    # Check for constraints (20 points)
    constraint_keywords = ["limit", "must", "should not", "under", "maximum", "only", "no more", "at most", "top"]
    if any(kw in prompt_lower for kw in constraint_keywords):
        score += 20

    return score


def apply_few_shot(task_description, examples, new_input):
    """Create a few-shot prompt by combining examples with a new input."""
    parts = [task_description, ""]

    for inp, out in examples:
        parts.append(f"Input: {inp}")
        parts.append(f"Output: {out}")
        parts.append("")

    parts.append(f"Input: {new_input}")
    parts.append("Output:")

    return "\n".join(parts)


# Test implementations
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
