import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime

st.set_page_config(page_title="Live Web Agent", layout="centered")

st.title("🌍 The Live Internet Agent")
st.write("Ask me anything about current events. I will browse the web to find the answer.")

# --- 1. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Config")
    user_api_key = st.text_input("Groq API Key:", type="password")

    # In the sidebar (with st.sidebar:), add a toggle variable:
    use_search = st.toggle("Enable Web Search", value=True)

    st.info("Equipped with: DuckDuckGo Web Search Tool")

# --- 2. THE MEMORY VAULT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. DRAW THE CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. THE CORE AGENTIC LOOP ---
if user_query := st.chat_input("Ask about today's news..."):

    if not user_api_key:
        st.error("Please enter your API Key in the sidebar.")
    else:
        # A. Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # B. Initialize the LangGraph Agent Engine
        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=user_api_key)
        web_tool = DuckDuckGoSearchRun()

        # --- Write a simple if/else logic block ---
        if use_search:
            active_tools = [web_tool]
        else:
            active_tools = [] # An empty list means no tools!

        # Pass active_tools into the create_react_agent function instead of [web_tool]
        agent = create_react_agent(llm, tools=active_tools)

        # C. THE BRIDGE: Translate Streamlit Memory -> LangGraph Memory
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Adjust the instructions so the AI knows if it can search or not
        if use_search:
            search_instruction = "You MUST use 'duckduckgo_search' to find current information."
        else:
            search_instruction = "Web search is DISABLED. Answer using your internal knowledge only."

        sys_prompt = (
            f"You are a live research assistant. Today's date is {current_date}. "
            f"You have access to exactly ONE tool: 'duckduckgo_search'. "
            f"{search_instruction}"
        )

        langgraph_history = [SystemMessage(content=sys_prompt)]

        for m in st.session_state.messages:
            if m["role"] == "user":
                langgraph_history.append(HumanMessage(content=m["content"]))
            else:
                langgraph_history.append(AIMessage(content=m["content"]))

        # D. Execute the Agent
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing..."):
                try:
                    result_state = agent.invoke({"messages": langgraph_history})
                    bot_answer = result_state["messages"][-1].content
                    st.markdown(bot_answer)
                    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
                except Exception as e:
                    st.error(f"The agent encountered an error: {e}")
