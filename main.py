import langchain
from chromadb import ChromaDB
import openai

def index_text_files(directory):
    chromadb = ChromaDB()
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                chromadb.insert(text)
    return chromadb

def get_user_prompt():
    return input("Enter your prompt: ")

def search_text(chromadb, prompt):
    results = chromadb.search(prompt)
    return results

def generate_response(prompt):
    # Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
    openai.api_key = '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'
    
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    return response.choices[0].text.strip()

def main():
    directory = 'txt'  # Directory containing the text files
    chromadb = index_text_files(directory)
    prompt = get_user_prompt()
    results = search_text(chromadb, prompt)
    if results:
        top_result = results[0]
        response = generate_response(top_result)
        print("OpenAI Response:")
        print(response)
    else:
        print("No results found.")

if __name__ == '__main__':
    main()
