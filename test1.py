from google.ai import generativelanguage as glm
import google.generativeai as genai

# Set up your API key
genai.configure(api_key='AIzaSyChW4wRNJ6A9kllmAWsI5L3BYgvFdfZ9wU')

# List available models
models = genai.list_models()
for model in models:
    print(model.name)