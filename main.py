import langchain
from whoosh import index, fields, qparser
from whoosh.index import create_in
from whoosh.qparser import QueryParser
import openai
import os

def index_text_files(directory):
    schema = fields.Schema(content=fields.TEXT(stored=True))
    if not os.path.exists("index"):
        os.mkdir("index")
    ix = create_in("index", schema)
    writer = ix.writer()
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                writer.add_document(content=text)
    writer.commit()
    return ix

def search_text(index, prompt):
    with index.searcher() as searcher:
        query = QueryParser("content", index.schema).parse(prompt)
        results = searcher.search(query)
        return results

def get_user_prompt():
    return input("Enter your prompt: ")

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
