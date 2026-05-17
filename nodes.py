import json
import asyncio
import os
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from state import AgentState
from mock_apis import fetch_weather_forecast, fetch_city_images, mock_web_search
from vector_store import query_vector_store


def _get_llm():
    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=groq_key,
        )
    elif anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.3,
            anthropic_api_key=anthropic_key,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=openai_key,
        )
    elif google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=google_key,
        )
    else:
        raise EnvironmentError("No API key found. Set GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_weather_forecast",
            "description": "Fetch a 7-day weather forecast for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_city_images",
            "description": "Fetch high-quality travel images for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"},
                    "count": {"type": "integer", "description": "Number of images", "default": 4},
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_REGISTRY: Dict[str, Any] = {
    "fetch_weather_forecast": fetch_weather_forecast,
    "fetch_city_images": fetch_city_images,
}


async def _execute_tool_call(tool_name: str, tool_args: dict) -> Any:
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    fn = TOOL_REGISTRY[tool_name]
    return await fn(**tool_args)


def extract_city_node(state: AgentState) -> dict:
    # memory fixing, if city state already saved then it wont be extracted again
    if existing_city and existing_city.lower() != "unknown" and existing_city.strip() != "":
        return {"city": existing_city}

    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        ""
    )

    # Check karo kya yeh followup hai
    followup_keywords = ["next week", "what about", "tomorrow", "forecast", "weather", "instead"]
    is_followup = any(kw in last_human.lower() for kw in followup_keywords)

    if is_followup and existing_city:
        return {"city": existing_city}

    llm = _get_llm()
    extraction_prompt = SystemMessage(content="""
You are a city name extractor. Extract ONLY the city name from the user message.
Respond with ONLY the city name, nothing else.
If no city is mentioned, respond with unknown.
Examples:
Tell me about Kyoto -> Kyoto
What is Paris like? -> Paris
I want to visit New York City -> New York
What about next week? -> unknown
""")
    response = llm.invoke([extraction_prompt, HumanMessage(content=last_human)])
    city = response.content.strip().strip('"').strip("'")
    return {"city": city}


def router_node(state: AgentState) -> dict:
    city = state.get("city", "")
    found, content = query_vector_store(city)
    if found:
        return {
            "use_vector_store": True,
            "city_summary": content,
            "source": "vector_store",
        }
    else:
        return {
            "use_vector_store": False,
            "source": "web_search",
        }


def vector_store_node(state: AgentState) -> dict:
    city = state.get("city", "")
    raw_content = state.get("city_summary", "")
    llm = _get_llm()
    summary_prompt = f"""
Using the following factual information about {city}, write a concise 3-4 paragraph
travel summary for a visitor. Focus on top attractions, best time to visit,
cultural highlights, and practical tips. Keep it engaging and informative.

Source material:
{raw_content}
"""
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return {"city_summary": response.content}


async def web_search_node(state: AgentState) -> dict:
    city = state.get("city", "")
    search_results = await mock_web_search(f"{city} travel guide city info")
    llm = _get_llm()
    summary_prompt = f"""
Based on these search results about {city}, write a helpful 3-4 paragraph
travel summary covering attractions, culture, best time to visit, and practical tips.

Search results:
{search_results}
"""
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return {"city_summary": response.content}


async def parallel_fetch_node(state: AgentState) -> dict:
    city = state.get("city", "")
    messages = list(state.get("messages", []))
    llm = _get_llm()

    tool_request_msg = HumanMessage(
        content=f"Please fetch the weather forecast and images for {city}. Use both available tools."
    )

    if hasattr(llm, 'bind_tools'):
        llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)
    else:
        llm_with_tools = llm

    ai_response = llm_with_tools.invoke(messages + [tool_request_msg])
    messages.append(ai_response)

    tool_calls = getattr(ai_response, 'tool_calls', []) or []

    if not tool_calls:
        weather, images = await asyncio.gather(
            fetch_weather_forecast(city),
            fetch_city_images(city),
        )
        return {
            "weather_forecast": weather,
            "image_urls": images,
            "messages": messages,
        }

    coroutines = []
    call_metadata = []

    for tc in tool_calls:
        if isinstance(tc, dict):
            tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
            raw_args = tc.get("args") or tc.get("function", {}).get("arguments", "{}")
            tool_id = tc.get("id", f"call_{tool_name}")
        else:
            tool_name = getattr(tc, "name", "")
            raw_args = getattr(tc, "args", {})
            tool_id = getattr(tc, "id", f"call_{tool_name}")

        if isinstance(raw_args, str):
            tool_args = json.loads(raw_args)
        else:
            tool_args = raw_args

        coroutines.append(_execute_tool_call(tool_name, tool_args))
        call_metadata.append((tool_id, tool_name))

    results = await asyncio.gather(*coroutines, return_exceptions=True)

    weather_data = None
    image_data = None

    for (tool_id, tool_name), result in zip(call_metadata, results):
        if isinstance(result, Exception):
            result_str = f"Error: {str(result)}"
        else:
            result_str = json.dumps(result)

        messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_id,
                name=tool_name,
            )
        )

        if tool_name == "fetch_weather_forecast" and not isinstance(result, Exception):
            weather_data = result
        elif tool_name == "fetch_city_images" and not isinstance(result, Exception):
            image_data = result

    if weather_data is None:
        weather_data = await fetch_weather_forecast(city)
    if image_data is None:
        image_data = await fetch_city_images(city)

    return {
        "weather_forecast": weather_data,
        "image_urls": image_data,
        "messages": messages,
    }


def aggregator_node(state: AgentState) -> dict:
    city = state.get("city", "Unknown")
    summary = state.get("city_summary", "No summary available.")
    weather = state.get("weather_forecast", [])
    images = state.get("image_urls", [])
    source = state.get("source", "unknown")

    final_response = {
        "city": city,
        "city_summary": summary,
        "weather_forecast": weather,
        "image_urls": images,
        "source": source,
    }

    assistant_msg = AIMessage(
        content=f"Here is the complete travel guide for {city}!"
    )

    return {
        "final_response": final_response,
        "messages": [assistant_msg],
    }


def route_by_knowledge(state: AgentState) -> str:
    if state.get("use_vector_store"):
        return "vector_store"
    return "web_search"
