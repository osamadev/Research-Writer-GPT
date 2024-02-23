import argparse
import asyncio
import json
import math
import sys

# asyncio.set_event_loop(asyncio.new_event_loop())

from millify import millify
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from trulens_eval.db_migration import MIGRATION_UNKNOWN_STR
from trulens_eval.ux.styles import CATEGORY

st.runtime.legacy_caching.clear_cache()

from trulens_eval import Tru, TruChain
from trulens_eval.ux import styles
from trulens_eval.ux.components import draw_metadata

from langchain_community.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import initialize_agent, AgentType
import pinecone
import os
import random
from utils.feedback_functions import load_feedback_functions
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser

st.set_page_config(page_title="Leaderboard", layout="wide")

if "authentication_status" not in st.session_state \
    or st.session_state["authentication_status"] == None or st.session_state["authentication_status"] == False:
    switch_page("Home")

from trulens_eval.ux.add_logo import add_logo_and_style_overrides

# add_logo_and_style_overrides()

database_url = None

from trulens_eval.tru_custom_app import instrument
from trulens_eval.tru_custom_app import TruCustomApp

class ResearchWriterApp:
    @instrument
    async def generate_response(self, prompt):
        from langchain_community.chat_models import ChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        openai_model = "gpt-4-0125-preview"

        llm = ChatOpenAI(model_name=openai_model, temperature=0.2)

        tools = load_tools()
        chat_agent = create_react_agent(
            llm=llm, tools=tools, prompt=build_prompts())

        agent_executor = AgentExecutor(agent=chat_agent, tools=tools, handle_parsing_errors=True)

        response = await agent_executor.ainvoke({'input': prompt})
        st.write(response)
        if 'output' in response:
            return response['output']
        return None

def streamlit_app():
    tru = Tru(database_url=database_url)
    # tru.reset_database()

    lms = tru.db
    # Set the title and subtitle of the app
    st.title("App Leaderboard")
    st.write(
        "Average feedback values displayed in the range from 0 (worst) to 1 (best)."
    )
    df, feedback_col_names = lms.get_records_and_feedback([])
    feedback_defs = lms.get_feedback_defs()
    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "") or
            row.feedback_json["implementation"]["name"]
        ): row.feedback_json.get("higher_is_better", True)
        for _, row in feedback_defs.iterrows()
    }

    with st.sidebar:
        if st.button("**Evaluate using Predefined Questions**"):
            filePath = './eval_questions.txt'
            eval_completions_from_questions(filePath)
            st.sidebar.success("Feedback functions have been executed successfully!")
            st.rerun()

    if df.empty:
        st.write("No records yet...")
        return

    df = df.sort_values(by="app_id")

    if df.empty:
        st.write("No records yet...")

    apps = list(df.app_id.unique())
    st.markdown("""---""")

    for app in apps:
        app_df = df.loc[df.app_id == app]
        if app_df.empty:
            continue
        app_str = app_df["app_json"].iloc[0]
        app_json = json.loads(app_str)
        metadata = app_json.get("metadata")
        # st.text('Metadata' + str(metadata))
        st.header(app, help=draw_metadata(metadata))
        app_feedback_col_names = [
            col_name for col_name in feedback_col_names
            if not app_df[col_name].isna().all()
        ]
        col1, col2, col3, col4, *feedback_cols, col99 = st.columns(
            5 + len(app_feedback_col_names)
        )
        latency_mean = (
            app_df["latency"].
            apply(lambda td: td if td != MIGRATION_UNKNOWN_STR else None).mean()
        )

        # app_df_feedback = df.loc[df.app_id == app]

        col1.metric("Records", len(app_df))
        col2.metric(
            "Average Latency (Seconds)",
            (
                f"{millify(round(latency_mean, 5), precision=2)}"
                if not math.isnan(latency_mean) else "nan"
            ),
        )
        col3.metric(
            "Total Cost (USD)",
            f"${millify(round(sum(cost for cost in app_df.total_cost if cost is not None), 5), precision = 2)}",
        )
        col4.metric(
            "Total Tokens",
            millify(
                sum(
                    tokens for tokens in app_df.total_tokens
                    if tokens is not None
                ),
                precision=2
            ),
        )

        for i, col_name in enumerate(app_feedback_col_names):
            mean = app_df[col_name].mean()

            st.write(
                styles.stmetricdelta_hidearrow,
                unsafe_allow_html=True,
            )

            higher_is_better = feedback_directions.get(col_name, True)

            if "distance" in col_name:
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta_color="normal"
                )
            else:
                cat = CATEGORY.of_score(mean, higher_is_better=higher_is_better)
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta=f"{cat.icon} {cat.adjective}",
                    delta_color=(
                        "normal" if cat.compare(
                            mean, CATEGORY.PASS[cat.direction].threshold
                        ) else "inverse"
                    ),
                )

        with col99:
            if st.button("Select App", key=f"app-selector-{app}"):
                st.session_state.app = app
                switch_page("TruLens Evaluations")

        st.markdown("""---""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_coversational_chain():
    from langchain_community.chat_models import ChatOpenAI
    index = initialize_pinecone()
    model_name = 'text-embedding-ada-002'
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

    openai_model = "gpt-4-0125-preview"

    text_field = "text"

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    retriever = vectorstore.as_retriever()
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name=openai_model, temperature=0.1)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def initialize_pinecone():
    index_name = "research-assistant" 

    from pinecone import Pinecone

    pc = Pinecone(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENV"]
        
    )
    # if index_name not in pc.list_indexes():
    #     pc.create_index(name=index_name, metric='cosine', dimension=1536, spec={})
    # Target the index
    index = pc.Index(index_name)
    return index

def load_fb_funcs(rag_chain):
    from trulens_eval.feedback.provider import OpenAI
    from trulens_eval import Feedback
    import numpy as np

    # Initialize provider class
    openai = OpenAI()
    feedbacks = []
    # select context to be used in feedback. the location of context is app specific.
    from trulens_eval.app import App
    context = App.select_context(rag_chain)

    from trulens_eval.feedback import Groundedness
    grounded = Groundedness(groundedness_provider=OpenAI())
    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect()) # collect context chunks into a list
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(openai.relevance).on_input_output()
    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(openai.qs_relevance)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )
    feedbacks.append(f_qa_relevance)
    feedbacks.append(f_context_relevance)
    feedbacks.append(f_groundedness)
    return feedbacks

def eval_completions_from_questions(filePath: str):
    tru = Tru()
    tru.reset_database()
    # Initialize TruChain with qa_chain and feedback functions
    (f_groundedness, f_qa_relevance, f_context_relevance, _, f_violent, _, _) = \
    load_feedback_functions()
    app = create_coversational_chain()
    feedback_funcs = load_fb_funcs(app)
    tru_recorder = TruChain(app,
                            app_id='Research Writer GPT',
                            feedbacks=feedback_funcs)
    
    # Read questions from a file
    with open(filePath, 'r') as file:
        questions = file.readlines()

    for question in questions[:15]:
        with tru_recorder as recording:
            app.invoke(question)

def eval_gpt4_completions(user_questions):
    tru = Tru()
    tru.reset_database()
    # Initialize TruChain with qa_chain and feedback functions
    (f_groundedness, f_qa_relevance, f_context_relevance, f_hate, f_violent, f_selfharm, f_maliciousness) = \
    load_feedback_functions()
    from langchain_community.chat_models import ChatOpenAI

    openai_model = "gpt-4-0125-preview"
    llm = ChatOpenAI(model_name=openai_model, temperature=0.2)
    tru_recorder = TruChain(llm,
                            app_id='Research Writer GPT',
                            feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness, f_violent])
    
    # Check if user questions list is empty
    if not user_questions:
        return False
    else:  
        # Randomly select up to three questions
        selected_questions = random.sample(user_questions, min(1, len(user_questions)))
        for question in selected_questions:
            tru_recorder(question)
        return True

def load_tools():
    from langchain.tools import Tool
    from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun, SemanticScholarAPIWrapper
    from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
    from langchain_community.tools import PubmedQueryRun
    from langchain_community.utilities.arxiv import ArxivAPIWrapper
    from langchain_community.utilities import SerpAPIWrapper

    semantic_scholar_tool = SemanticScholarQueryRun(name="Semantic Scholar", 
                                                    api_wrapper=SemanticScholarAPIWrapper(top_k_results = 10, load_max_docs = 10),
                                                    description="""Use this tool to answer questions about sentific papers and literatures,
                                     or to provide recommendations or summaries about the papers or research topics. """, )
    
    google_scholar_tool = Tool("Google Scholar", func=GoogleScholarAPIWrapper().run, description="""Use this tool to help answering 
                                                reasrch questions and provide recommendations on the research papers and literatures.
                                                If the researcher asked for links to the papers or literatures, use this search tool first.""")

    pubmed_research_tool = PubmedQueryRun()
    
    arxiv_tool = Tool("Arxiv Articles and Literatures", 
                      ArxivAPIWrapper(top_k_results = 10,
            ARXIV_MAX_QUERY_LENGTH = 300,
            load_max_docs = 10,
            load_all_available_meta = False,
            doc_content_chars_max = 40000).run,
                      description="Use this tool to get details about articles published in Arxiv"
                      )

    search = SerpAPIWrapper()

    repl_tool = Tool(
    name="google_search",
    description="Use this tool to search for more details to enrich the final response, and to provide grounded resources and references for the papers and articles.",
    func=search.run)
    
    tools = [semantic_scholar_tool, google_scholar_tool, repl_tool]
    return tools

def build_prompts():
    with open('prompts/main.txt', 'r') as file:
        content = file.read()

    prompt: PromptTemplate = PromptTemplate(
        input_variables=["input"],
        template=content)
    return prompt

# Define the main function to run the app
def main():
    streamlit_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

    database_url = args.database_url
    
    main()
