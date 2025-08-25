import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result


class Chatbot:
    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow-up entry: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """You are an expert data analyst and conversational assistant specializing in CSV data analysis.

Your role is to provide accurate, insightful, and actionable answers based on the provided CSV data context.

CRITICAL INSTRUCTIONS FOR NUMERICAL QUERIES:
1. When asked for maximum, minimum, or highest/lowest values, carefully examine ALL data points in the context
2. Always double-check numerical comparisons - don't assume the first match is correct
3. For "highest" or "maximum" queries, scan through ALL values to find the true maximum
4. Quote exact numbers from the data - never approximate or guess

GENERAL INSTRUCTIONS:
1. Analyze the context carefully to understand the data structure, columns, and relationships
2. Provide specific, data-driven answers with exact values, percentages, or trends when possible
3. If the context doesn't contain sufficient information, clearly state what's missing
4. Use the actual column names and values from the data in your responses
5. When appropriate, suggest follow-up questions or additional analyses
6. Format numerical data clearly (use commas for thousands, appropriate decimal places)
7. Respond in the same language as the user's question

CONTEXT DATA:
{context}

USER QUESTION: {question}

RESPONSE GUIDELINES:
- Be specific and cite actual data points
- For numerical queries, verify your answer by checking all relevant data points
- Explain any calculations or reasoning
- Highlight key insights or patterns
- If data is incomplete, suggest what additional information would be helpful
- Use clear, professional language while remaining conversational

Answer:"""

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    def _estimate_tokens(self, text, model_name="gpt-3.5-turbo"):
        """
        Estimate token count for a given text
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except:
            # Fallback estimation: roughly 4 characters per token
            return len(text) // 4

    def _truncate_history_if_needed(self, query, max_tokens=250000):
        """
        Truncate chat history if the total estimated tokens exceed the limit
        """
        # Estimate tokens for current query
        query_tokens = self._estimate_tokens(query)

        # Estimate tokens for history
        history_text = ""
        for q, a in st.session_state.get("history", []):
            history_text += f"Q: {q}\nA: {a}\n"

        history_tokens = self._estimate_tokens(history_text)
        total_tokens = query_tokens + history_tokens

        # If exceeding limit, truncate history
        if total_tokens > max_tokens:
            # Keep only the last few exchanges
            max_history_items = 5
            if len(st.session_state["history"]) > max_history_items:
                st.session_state["history"] = st.session_state["history"][-max_history_items:]
                st.warning(f"Chat history truncated to prevent token limit issues. Keeping last {max_history_items} exchanges.")

    def _get_enhanced_retriever(self):
        """
        Create an optimized retriever with responsive parameters
        """
        try:
            # Force similarity search to avoid MMR dimension issues
            search_type = "similarity"  # Always use similarity to avoid dimension mismatches
            num_docs = st.session_state.get("num_docs", 8)  # Increase to 8 for better context

            search_kwargs = {"k": num_docs}

            # Test the vector store with a simple query first
            try:
                test_results = self.vectors.similarity_search("test", k=1)
                if not test_results:
                    raise ValueError("Vector store returned no results")
            except Exception as e:
                st.error(f"Vector store test failed: {str(e)}")
                raise ValueError(f"Vector store is corrupted: {str(e)}")

            return self.vectors.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
        except Exception as e:
            st.error(f"Retriever creation failed: {str(e)}")
            raise e

    def conversational_chat(self, query):
        """
        Optimized conversational chat with timeout protection
        """
        try:
            # Check and truncate history if needed
            self._truncate_history_if_needed(query)

            # Use conservative model settings for responsiveness
            model_params = {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": 800,  # Reduce for faster responses
                "request_timeout": 30,  # Add timeout
            }

            # Model-specific optimizations
            if "gpt-4" in self.model_name:
                model_params["max_tokens"] = 1000
                model_params["request_timeout"] = 45

            llm = ChatOpenAI(**model_params)

            # Simplified retrieval chain for speed
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
                retriever=self._get_enhanced_retriever(),
                return_source_documents=False,  # Disable for speed
                verbose=False,
                combine_docs_chain_kwargs={"prompt": self.QA_PROMPT}
            )

            chain_input = {"question": query, "chat_history": st.session_state["history"]}

            # Add progress indicator
            with st.spinner("Analyzing your data..."):
                result = chain(chain_input)

            answer = result["answer"]
            st.session_state["history"].append((query, answer))

            # Simplified token counting
            try:
                count_tokens_chain(chain, chain_input)
            except:
                pass  # Don't let token counting break the response

            return answer

        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg:
                st.error("Request timed out. Try a simpler question or check your connection.")
                return "The request took too long to process. Please try asking a simpler question."
            elif "tokens" in error_msg and "max" in error_msg:
                st.error("Token limit exceeded. Try asking a more specific question.")
                return "Your request exceeded the token limit. Please ask a more specific question."
            elif "rate" in error_msg or "quota" in error_msg:
                st.error("API rate limit reached. Please wait and try again.")
                return "Rate limit reached. Please wait a moment and try again."
            else:
                st.error(f"Error: {str(e)}")
                return "An error occurred. Please try rephrasing your question or restart the chat."
