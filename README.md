# Multi-Modal Travel Assistant
AI Engineering Assignment — Digital Alpha Platforms
A LangGraph-powered travel agent that basically takes a city name and returns a  very rich and interactive travel guide — complete with a city summary, 7-day weather forecast, and photo gallery.All rendered in a Streamlit web app.

# Architecture Overview:
The system is built as a state machine using LangGraph. Every user query is followed through a pipeline of 5 nodes, where eachnode is  responsible for exactly one thing.
[extract_city] → [router] → [vector_store] or [web_search] → [parallel_fetch] → [aggregator]

extract_city then reads the user's natural language input and search for the city name using the LLM.

router then checks whether that city exists in the local ChromaDB vector store or not. If yes,then it routes left and if no, it routes right. The conditional edge that drives the intelligent switching behavior.

vector_store work is to pre-loaded facts about Paris, Tokyo, or New York and finally generates a clean travel summary for the user using the LLM.

web_search work is to handle unknown cities by calling a mock search function and summarizing the results.

parallel_fetch fires the weather API and image API at the same time using asyncio.gather (not sequentially). Both results come back together and are then appended to state.

aggregator takes everything (summary, forecast, images ) and packages it into a structured Pydantic object that Streamlit parses to render the final UI.

# Tech Stack

Orchestration: LangGraph with typed AgentState
Vector Store: ChromaDB (in-memory, EphemeralClient)
LLM: Claude 3.5 Sonnet (Anthropic) / GPT-4o (OpenAI)
Frontend: Streamlit with Plotly charts
Structured Output: Pydantic models
Async: Python asyncio for parallel API calls


# The Three Distinction Challenges
Distinction 1 — Manual Tool Execution
I didn’t leverage any prebuilt functions like prebuilt.ToolNode or create_tool_calling_agent(). Rather, in parallel_fetch_node(), I manually parsed the LLM’s tool_calls object, iterated through it to execute functions on my own and appended the resulting messages as a ToolMessage() object in the state. This proves my understanding of how an LLM calls tools internally..
Distinction 2 — Parallel Fan-Out
Data od weather and pictures of the cities are totally unrelated. Thus, instead of retrieving them in sequence, I execute both coroutines concurrently using asyncio.gather. When working with actual APIs in production, this reduces the waiting time by about half.
Distinction 3 — Memory and Context Preservation
The checkpointer I implemented for LangGraph's MemorySaver was done with a new thread_id for each session. So, in case the user asks about Tokyo and says "What about next week?" after that, the chatbot accesses the saved state, then identifies the known city, and continues the conversation accordingly.

# How to Run
1. Install dependencies
bashpip install -r requirements.txt
2. Set your API key
bash# Mac/Linux
export ANTHROPIC_API_KEY=your_key

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_key"
--- Run the app
bashstreamlit run app.py
 Generate graph diagram
bashpython graph.py

# Routing Logic
Three cities are hardcoded into the ChromaDB vector database: Paris, Tokyo, and New York. If the user queries any of these three, the agent fetches this information locally without requiring a web search at all. For any other city name, the agent will always route via the mocked web search endpoint.

Mock APIs
As the emphasis of this assignment is architecture and not API functionality, I have created some mock methods to simulate realistic latencies and return meaningful JSON objects. Substituting actual OpenWeatherMap or Unsplash API tokens would be as easy as changing a single line in each method.
Placeholders are provided for unknown city images, which would in reality be handled by the Flickr API or Unsplash API for photographs of particular cities.

Project Structure
travel_assistant/
├── app.py              Streamlit frontend
├── graph.py            LangGraph topology and compilation
├── nodes.py            All node functions
├── state.py            Typed AgentState and Pydantic models
├── vector_store.py     ChromaDB setup and query logic
├── mock_apis.py        Simulated weather and image APIs
├── config.py           Environment variable loader
├── requirements.txt    Dependencies
└── graph.png           Auto-generated graph topology diagram
