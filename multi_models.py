import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from pydub import AudioSegment
from pydub.playback import play
from first_tool import tools, handle_tool_call

load_dotenv(override=True)

MODEL = 'gpt-4o-mini'
openai = OpenAI()


system_message = """You are a helpful assistant for an Airline called FlightAI
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so."""

def artist(city):
    image_response = openai.images.generate(
        model='dall-e-3',
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
        response_format="b64_json"
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)

def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]
    talker(reply)
    return history, image


with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")
    
    def do_entry(message, history):
        history += [{"role": "user", "content": message}]
        return "", history
    
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)