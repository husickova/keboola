from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from github import Github
from github import GithubException
import os

def get_sha_for_tag(repository, tag):
    """
    Returns a commit PyGithub object for the specified repository and tag.
    """
    branches = repository.get_branches()
    matched_branches = [match for match in branches if match.name == tag]
    if matched_branches:
        return matched_branches[0].commit.sha

    tags = repository.get_tags()
    matched_tags = [match for match in tags if match.name == tag]
    if not matched_tags:
        raise ValueError('No Tag or Branch exists with that name')
    return matched_tags[0].commit.sha


def download_directory(repository, sha, server_path):
    """
    Download all contents at server_path with commit tag sha in
    the repository.
    """
    if os.path.exists(server_path):
        shutil.rmtree(server_path)

    os.makedirs(server_path)
    contents = repository.get_dir_contents(server_path, ref=sha)

    for content in contents:
        print ("Processing %s" % content.path)
        if content.type == 'dir':
            os.makedirs(content.path)
            download_directory(repository, sha, content.path)
        else:
            try:
                path = content.path
                file_content = repository.get_contents(path, ref=sha)
                file_data = base64.b64decode(file_content.content)
                file_out = open(content.path, "w+")
                file_out.write(file_data)
                file_out.close()
            except (GithubException, IOError) as exc:
                print('Error processing %s: %s', content.path, exc)

def usage():
    """
    Prints the usage command lines
    """
    print("usage: gh-download --token=token --org=org --repo=repo --branch=main --folder=\"soubory\"")

def main(argv):
    """
    Main function block
    """
    try:
        opts, args = getopt.getopt(argv, "t:o:r:b:f:", ["token=", "org=", "repo=", "branch=", "folder="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-t", "--token"):
            token = arg
        elif opt in ("-o", "--org"):
            org = arg
        elif opt in ("-r", "--repo"):
            repo = arg
        elif opt in ("-b", "--branch"):
            branch = arg
        elif opt in ("-f", "--folder"):
            folder = arg

    github = Github(token)
    organization = github.get_organization(org)
    repository = organization.get_repo(repo)
    sha = get_sha_for_tag(repository, branch)
    download_directory(repository, sha, folder)

if __name__ == "__main__":
    """
    Entry point
    """
    main(sys.argv[1:])

#Step 05: Split the Extracted Data into Text Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

#Step 06:Downlaod the Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Step 08: Create Embeddings for each of the Text Chunk
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

#Import Model
llm = LlamaCpp(
    streaming = True,
    model_path = "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4096
)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}))

query = "Based on the information you have, who would be strong candidates for open positions at Keboola and why?"

qa.invoke(query)

import sys

while True:
  user_input = input(f"Input Prompt: ")
  if user_input == 'exit':
    print('Exiting')
    sys.exit()
  if user_input == '':
    continue
  result = qa({'query': user_input})
  print(f"Answer: {result['result']}")

import streamlit as st

prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")

st.title("Keboola Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
