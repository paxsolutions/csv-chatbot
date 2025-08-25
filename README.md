# CSV Chatbot 🤖

### An AI chatbot featuring conversational memory, designed to enable users to discuss their CSV data in a more intuitive manner. 📄
By integrating the strengths of Langchain and OpenAI, CSV Chatbot employs large language models to provide users with seamless, context-aware natural language interactions for a better understanding of their CSV data.🧠

## Running Locally 💻
Follow these steps to set up and run the service locally :

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
Clone the repository :

Navigate to the project directory :

`cd csv-chatbot`

Create a virtual environment :
```bash
python -m venv .venv/chatbot && source .venv/chatbot/bin/activate
```

Install the required dependencies in the virtual environment :

`pip install -r requirements.txt`

Launch the chat service locally :

`streamlit run src/chatbot_csv.py --server.port=8501`

#### That's it! The service is now up and running locally. 🤗

## Docker 🐋

Navigate to the project directory :

`cd csv-chatbot`

Build the Docker image :

`docker-compose build`

Run the Docker container :

`docker-compose up`

## Information 📝:
CSV Chatbot features a chatbot with memory and a CSV agent. The chatbot is specialized in discussing unique elements within the CSV with the user in a friendly and conversational manner.
