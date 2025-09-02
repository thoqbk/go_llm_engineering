from create_brochure import get_brochure_prompts, openai, MODEL
import gradio as gr

def stream_brochure(company_name, url):
    system_prompt, user_prompt = get_brochure_prompts(company_name, url)
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_prompt}
        ],
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

view = gr.Interface(
    fn=stream_brochure,
    inputs=[
        gr.Textbox(label="Company Name:", value="Hugging Face"),
        gr.Textbox(label="URL:", value="https://huggingface.co/"),
    ],
    outputs=[gr.Markdown(label="Brochure:")],
    flagging_mode="never"
)
view.launch()