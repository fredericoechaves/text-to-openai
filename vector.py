import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()
openai_api = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
directory = "txt"
chroma = Chroma(embedding_function=embeddings, persist_directory="db")

def embed_text_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                ids = chroma.add_texts([text], ids=[filename], metadatas=[{'name': filename}])
                print(ids)
    chroma.persist()

def get_user_prompt():
    return input("Enter your prompt: ")

def main():
    embed_text_files(directory)
    retriever = chroma.as_retriever(search_kwargs={"k": 3})
    retriever.search_kwargs
    while True:
        prompt = get_user_prompt()
        if prompt.lower() in ['quit', 'exit']:
            break
        qa_chain = RetrievalQA.from_chain_type(llm=openai_api, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
        response = qa_chain(prompt)
        print(response['result'])
        for result in response["source_documents"]:
            print(result.metadata)

if __name__ == '__main__':
    main()