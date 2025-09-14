import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)

MODEL = "gpt-4o-mini"
openai = OpenAI()

ticket_prices = {"london": 100, "paris": 120, "new york": 200, "tokyo": 300, "sydney": 250, "singapore": 300, "hanoi": 300}

def get_ticket_price(destination_city):
    city = destination_city.lower()
    return ticket_prices.get(city, 'Unknown')

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is the ticket to the city?'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

tools = [{"type": "function", "function": price_function}]

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    return {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": price}),
        "tool_call_id": tool_call.id
    }, city

system_message = """You are a helpful assistant for an Airline called FlightAI
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so."""

def chat_with_tools(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

if __name__ == "__main__":
    gr.ChatInterface(fn=chat_with_tools, type="messages").launch()