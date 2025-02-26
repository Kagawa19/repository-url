from google.ai import generativelanguage as glm
import google.generativeai as genai

# Set up your API key
genai.configure(api_key='AIzaSyChW4wRNJ6A9kllmAWsI5L3BYgvFdfZ9wU')


# Choose a model from the list
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Generate content
response = model.generate_content("Write a short story about messi and ronaldo")

# Print the response
print(response.text)