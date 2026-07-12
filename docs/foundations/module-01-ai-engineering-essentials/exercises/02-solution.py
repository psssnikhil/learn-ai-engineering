from openai import OpenAI
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def chat(message: str) -> str:
    """
    Send a message to GPT and get a response
    
    Args:
        message: The user's message
        
    Returns:
        The AI's response
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant specialized in teaching AI engineering."},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Test your chatbot!
if __name__ == "__main__":
    print("Chatbot: Hello! I'm your AI assistant.")
    print("-" * 50)
    
    # Test with multiple questions
    questions = [
        "What is AI engineering?",
        "Explain it in one sentence.",
        "What skills do I need?"
    ]
    
    for question in questions:
        print(f"\nYou: {question}")
        response = chat(question)
        print(f"AI: {response}")

# Bonus: Add conversation memory
class ChatbotWithMemory:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
    
    def chat(self, message: str) -> str:
        # Add user message to history
        self.messages.append({"role": "user", "content": message})
        
        # Get response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

# Test with memory
print("\n" + "="*50)
print("CHATBOT WITH MEMORY")
print("="*50)

bot = ChatbotWithMemory()
print("You: My name is Alex")
print(f"AI: {bot.chat('My name is Alex')}")
print("\nYou: What's my name?")
print(f"AI: {bot.chat('What is my name?')}")

