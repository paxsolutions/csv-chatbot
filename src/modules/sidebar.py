import streamlit as st


class Sidebar:
    MODEL_OPTIONS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 1.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about():
        about = st.sidebar.expander("About 🤖")
        sections = [
            "#### Chatbot is an AI chatbot featuring conversational memory, designed to enable users to discuss their CSV data in a more intuitive manner. 📄",
            "#### It employs large language models to provide users with seamless, context-aware natural language interactions for a better understanding of their CSV data. 🌐",
            "#### Powered by [Langchain](https://github.com/hwchase17/langchain), [OpenAI](https://platform.openai.com/docs/models/o4-mini) and [Streamlit](https://github.com/streamlit/streamlit) ⚡",
        ]
        for section in sections:
            about.write(section)

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        model = st.selectbox(
            label="Model",
            options=self.MODEL_OPTIONS,
            help="GPT-4o-mini: Fast and cost-effective. GPT-4o: Latest model with best performance. GPT-4-turbo: Good balance of speed and quality."
        )
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
            help="Lower values (0.0-0.3) for factual analysis, higher values (0.4-0.8) for creative insights"
        )
        st.session_state["temperature"] = temperature

    def retrieval_settings(self):
        st.subheader("🔍 Retrieval Settings")

        search_type = st.selectbox(
            "Search Strategy",
            ["mmr", "similarity"],
            index=0,
            help="MMR provides diverse results, similarity focuses on most relevant matches"
        )
        st.session_state["search_type"] = search_type

        num_docs = st.slider(
            "Documents to Retrieve",
            min_value=3,
            max_value=15,
            value=5,
            help="More documents provide better context but may slow responses"
        )
        st.session_state["num_docs"] = num_docs

    def csv_agent_button(self):
        st.session_state.setdefault("show_csv_agent", False)
        if st.sidebar.button("CSV Agent"):
            st.session_state["show_csv_agent"] = not st.session_state["show_csv_agent"]

    def show_options(self):
        with st.sidebar.expander("🛠️ Tools", expanded=False):
            self.reset_chat_button()
            self.csv_agent_button()

        with st.sidebar.expander("⚙️ Model Settings", expanded=True):
            self.model_selector()
            self.temperature_slider()

        with st.sidebar.expander("🔍 Advanced Settings", expanded=False):
            self.retrieval_settings()

        st.session_state.setdefault("model", self.MODEL_OPTIONS[0])
        st.session_state.setdefault("temperature", self.TEMPERATURE_DEFAULT_VALUE)
        # May also use "mmr" for more diverse results
        st.session_state.setdefault("search_type", "similarity")
        st.session_state.setdefault("num_docs", 5)
