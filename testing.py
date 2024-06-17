import streamlit as st

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
import os
from typing import Literal
from langgraph.graph import Graph

from langchain_community.document_loaders import PyMuPDFLoader
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import OpenAIEmbeddings
import subprocess
from langchain_core.runnables.graph import  MermaidDrawMethod
from IPython.display import display, HTML, Image
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails,LLMRails
from openai import OpenAI
from dotenv import dotenv_values

# Load environment variables from .env file
env_vars = dotenv_values(".env")

# Set environment variables
os.environ['OPENAI_API_KEY'] = env_vars['OPENAI_API_KEY']
os.environ['NVIDIA_API_KEY'] = env_vars['NVIDIA_API_KEY']

model = ChatOpenAI(model="gpt-4o")
model2 = ChatNVIDIA(model="meta/llama3-70b-instruct")
client = OpenAI()


AgentState = {}
AgentState["messages"] = []
AgentState["codinglanguage"] = ""
AgentState["testingframework"] = ""
AgentState["testingcode"] = ""
AgentState["output"] = ""
AgentState["code"] = ""
gretriever = None

@st.cache_resource
def encoder():
    loader = PyMuPDFLoader("./Pytest.pdf")
    pages = loader.load_and_split()
    documents = []
    for page in pages:
        documents.append(page.page_content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        documents,
        embeddings
    )
    vectorstore
    retriever = vectorstore.as_retriever()
    return retriever
    

gretriever = encoder()



def parser(input):
    messages = input["messages"]
    user_input = messages[-1]
    query = "Your job is to parse the code exactly as it is provided from the user query without making syntax changes exept for indentation and line breaks as well and just print that. The following is the user query: "+user_input
    query2 = "Your job is to determine the testing tool/framework from the user query and just print the testing tool/framework and nothing else. If there it is not specified print NONE. The following is the user query: "+user_input
    response = model2.invoke(query)
    response2 = model2.invoke(query2)
    input["messages"].append(response.content) 
    input["testingframework"] = response2.content
    input["code"] = response.content
    print("Parsed")
    return input

def code_comp(input):
    messages = input["messages"]
    user_input = messages[-1]
    query = "Assume all imports are working and if it is just a function there is not need to have a class. Your job is to check if the logic of the code from the user query will compile if it can say YES and if not say NO. The following is the user query: "+user_input
    query2 = "Your job is to determine the coding language from the user query and just print the coding language and nothing else. The following is the user query: "+user_input
    response = model.invoke(query)
    response2 = model.invoke(query2)
    input["messages"].append(response.content+" "+response2.content)
    input["codinglanguage"] = response2.content
    print("Normal Code compiled")
    return input

def router(input):
    messages = input["messages"]
    user_input = messages[-1]
    print(user_input)
    if "NO" in user_input:
        return "fail_code_comp"
    else:
        return "Test Generator"
    
def router2(input):
    print("Reached the router")
    tFramework = input["testingframework"]
    if tFramework.lower() == "pytest":
        return "Test Compiler"
    else:
        return "__end__"
    

def test_generate(input):
    messages = input["messages"]
    user_input = messages[0]
    query = "Your job is to create test cases without any coding just purely logical for the user query. The following is the user query: "+user_input
    response = model.invoke(query)
    input["messages"].append(response.content)
    print("Tests Generated")
    return input

def test_to_code(input):
    messages = input["messages"]
    lang = input["codinglanguage"]
    tFramework = input["testingframework"]
    code = input["code"]
    user_input = messages[-1]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "context:\n<Documents>\n{context}\n</Documents>. ",
            ),
            ("user", "Make sure not to inclue any wrappers like ``` or ```python or additional explanation texts just raw code. {question}"),
        ]
    )
    chain = (
        {"context": gretriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    if tFramework != "NONE":
        query = query = """
        The following is the user query: """+user_input+"""Here is the code that the user provided:"""+code+"""Your job is to code test 
        cases in """+lang+""" for the code for the user query in """+tFramework+""". Make sure not to inclue any wrappers like
         ``` or ```python or additional explanation texts just raw code. Do not expect any importable files or modules to be provided 
         so you must put the users code into the program yourself and any corresponding imports needed in your program. 
        """
        response = chain.invoke(query)
    else:
        query = "Your job is to code only the test cases for "+lang+" for the code from the user query with. The following is the user query: "+user_input
        response = model2.invoke(query)
        
    print(tFramework)
    input["testingcode"] = response
    print("Code Generated")
    return input

def fail_code_comp(input):
    messages = input["messages"]
    response = "The code you provided will not compile. Please provide a valid code that can compile to generate tests."
    input["messages"].append(response)
    return input



def test_compiler(input):
    testCode = input["testingcode"]
    def run_code_and_store_output(code):
        # Write the code to a temporary Python file
        with open('test_code.py', 'w') as code_file:
            code_file.write(code)

        # Run the Python file and capture the output
        result = subprocess.run(['pytest', 'test_code.py'], capture_output=True, text=True)

        # Store the output in output.txt
        with open('output.txt', 'w') as output_file:
            output_file.write(result.stdout)
            if result.stderr:
                output_file.write('\nErrors:\n')
                output_file.write(result.stderr)
    run_code_and_store_output(testCode)
    print("Ran the code")
    return input

def test_output(input):
    # Open the file in read mode
    with open('output.txt', 'r') as file:
        # Read the contents of the file
        file_contents = file.read()

    # Store the contents in a string variable
    text_string = file_contents

    # Print the string to verify
    input["output"] = text_string
    print("stored the output")
    return input

    
    

def generate_response(input_text):
    # Create a new graph
    workflow = Graph()  

    workflow.add_node("Parser", parser)
    workflow.add_node("Code Compiler", code_comp)
    workflow.add_node("Test Generator", test_generate)
    workflow.add_node("Test to Code", test_to_code)
    workflow.add_node("fail_code_comp", fail_code_comp)
    workflow.add_node("Test Compiler", test_compiler)
    workflow.add_node("Test Output", test_output)

    workflow.add_edge("Parser", "Code Compiler")
    workflow.add_conditional_edges("Code Compiler", router)
    workflow.add_edge("Test Generator", "Test to Code")
    workflow.add_edge("fail_code_comp", "__end__")
    workflow.add_conditional_edges("Test to Code", router2)
    workflow.add_edge("Test Compiler", "Test Output")
    workflow.add_edge("Test Output","__end__")

    workflow.set_entry_point("Parser")

    app = workflow.compile()

    st.image(
    app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    ))

    inputs = {"messages": [input_text]}

    answer = app.invoke(inputs)
    
    # st.info(answer["messages"][-1])
    # st.info("The coding language you used is: "+answer["codinglanguage"])
    # st.info("The testing framework you used is: "+answer["testingframework"])

    # if answer["testingframework"].lower() == "pytest" and answer["output"] is not None:
    #     st.info("The testing output is :\n"+ answer["output"])
    
    return answer
  
st.title('🧪 Test Generating App')
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    


if st.session_state.messages == []:
    st.session_state.messages.append({"role": "assistant", "content": "Hello, I am here to assist you in creating test cases. Please provide some code and I will try to help you. I am mainly proficient in Python"})

for message in st.session_state.messages:
     with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    if prompt != "":
        # Display assistant response in chat message container
        with st.chat_message("user"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(prompt)

        prompt2 = "Here is the current query: "+prompt+"\nIf the current query contains CODE and asks to create TESTS say 'TRUE' and nothing else otherwise print 'FALSE' and nothing else."
        test = model.invoke(prompt2)
        print(test.content)
        if "TRUE" in test.content:
            with st.chat_message("assistant"):
                answer = generate_response(prompt)
                text = answer["messages"][-1]+"\n\nThe coding language you used is: "+answer["codinglanguage"]+"\n\nThe testing framework you used is: "+answer["testingframework"]
                if answer["testingframework"].lower() == "pytest" and answer["output"] is not None:
                    text += "\n\nThe testing output is :\n"+ answer["output"]
                st.markdown(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
        else:
            with st.chat_message("assistant"):
                st.markdown("Sorry that question is beyond the scope of what I can do")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry that question is beyond the scope of what I can do"})


    
    # Display user message in chat message container

# with st.form('my_form'):
#   text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#   submitted = st.form_submit_button('Submit')
#   if submitted:
#     generate_response(text)