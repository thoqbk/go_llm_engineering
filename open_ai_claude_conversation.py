from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

openai = OpenAI()
claude = anthropic.Anthropic()

gpt_model = 'gpt-4.1-mini'
claude_model = 'claude-3-5-haiku-latest'

gpt_system_prompt = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system_prompt = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting. Try to highlight the quotes in the other person's messages in the right previous message"

gpt_messages = ["Hi there"]
claude_messages = []

def call_claude():
    messages = [{"role": "user", "content": gpt_messages[-1]}]
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": claude_message})
        messages.append({"role": "user", "content": gpt})
    message = claude.messages.create(
        model=claude_model,
        system=claude_system_prompt,
        messages=messages,
        max_tokens=500
    )
    return message.content[0].text

def call_gpt():
    messages = [{"role": "system", "content": gpt_system_prompt}]
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
    completion = openai.chat.completions.create(
        model=gpt_model,
        messages=messages
    )
    return completion.choices[0].message.content

print(f"GPT:\n{gpt_messages[0]}\n")

for i in range(5):
    claude_next = call_claude()
    print(f"Claude:\n{claude_next}\n")
    claude_messages.append(claude_next)

    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)    
