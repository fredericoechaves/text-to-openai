import os
import openai
from dotenv import load_dotenv

def concatenate_text_files(directory):
    concatenated_text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                concatenated_text += text
    return concatenated_text

def get_user_prompt():
    return input("Enter your prompt: ")

def generate_response(prompt, text):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=text + prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    return response.choices[0].text.strip()

def main():
    directory = 'txt'  # Directory containing the text files
    concatenated_text = concatenate_text_files(directory)
    while True:
        prompt = get_user_prompt()
        if prompt.lower() in ['quit', 'exit']:
            break
        response = generate_response(prompt, concatenated_text)
        print("OpenAI Response:")
        print(response)

if __name__ == '__main__':
    main()