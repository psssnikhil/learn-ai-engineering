# TODO: Import OpenAI library
# from openai import OpenAI

# TODO: Initialize the client
# client = OpenAI()

def chat(message: str) -> str:
    """
    Send a message to GPT and get a response
    
    Args:
        message: The user's message
        
    Returns:
        The AI's response
    """
    # TODO: Create a chat completion
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful AI assistant."},
    #         {"role": "user", "content": message}
    #     ]
    # )
    
    # TODO: Extract and return the response
    # return response.choices[0].message.content
    
    pass  # Remove this line when you implement

# Test your chatbot!
if __name__ == "__main__":
    print("Chatbot: Hello! I'm your AI assistant.")
    
    # TODO: Test with a question
    response = chat("What is AI engineering?")
    print(f"AI: {response}")

