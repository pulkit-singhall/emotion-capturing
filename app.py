import torch # type: ignore
from transformers import pipeline # type: ignore
import gradio as gr # type: ignore
from dotenv import load_dotenv # type: ignore
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_dotenv()

task = os.getenv('task')
model = os.getenv('model')
gitHubLink = 'https://github.com/pulkit-singhall'
linkedInLink = 'https://www.linkedin.com/in/pulkit-singhal-a8113822a/'

pipe = pipeline(task, model, device=device, framework='pt')


def classify_text(text, top_results):
    result = pipe(text, top_k = top_results) # array of dict
    output = result[0]['label']
    output = output[:1].upper() + output[1:]

    for i in range(1,len(result)):
        label = result[i]['label']
        label = label[:1].upper() + label[1:]
        output = '\n'.join([output, label])
    return output


with gr.Blocks() as app:
    gr.Markdown(value = 'Get emotions from any textual sentence...', height = 28)
    with gr.Row():
        with gr.Column():
            sentence = gr.Textbox(label="English Sentence Here")
            slider = gr.Slider(
                label = 'Top Results you want', value = 1, 
                minimum = 1, maximum = 10, step = 1)
            with gr.Row():
                clear_btn = gr.ClearButton(components = [sentence], variant = 'secondary')
                classify_btn = gr.Button(value="Submit", variant = 'primary')
        with gr.Column():
            result = gr.Textbox(label="Required Emotions")

    classify_btn.click(classify_text, inputs=[sentence, slider], outputs=[result])
    examples = gr.Examples(examples=["That movie was amazing but I did not like the actors.", 
                                     "Helen is a good swimmer.",
                                     'What do you think about Elon Musk and his accomplishments?',
                                     'Today was a horrible day'],
                           inputs=[sentence])
    gr.Markdown(value = f'Check out my [GitHub]({gitHubLink}) and [LinkedIn]({linkedInLink})', 
                height = 28)

app.launch(share = True)