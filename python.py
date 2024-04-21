from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llm import generate_text

app = Flask(__name__)

# Load the model and tokenizer
model_name = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Function to generate text based on user input
def generate_text(prompt):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']

@app.route('/', methods=['GET', 'POST'])
def home():
   if request.method == 'POST':
        prompt = request.form['prompt']
        dataset_name = request.form.get('dataset_name', 'mlabonne/guanaco-llama2-1k')  # use default if not provided
        generated_text = generate_text(prompt, dataset_name)
        return render_template('output.html', prompt=prompt, generated_text=generated_text)
   return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True)
