"""
state.py — Typed LangGraph State for the Multi-Modal Travel Assistant

All nodes read from and write to this shared state object.
Using TypedDict keeps things typed without Pydantic overhead at the graph level.
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# Final structured output that Streamlit actually can renders

class WeatherDay(BaseModel):
    date: str
    temp_high: float
    temp_low: float
    condition: str
    humidity: int


class TravelResponse(BaseModel):
    city: str
    city_summary: str
    weather_forecast: List[WeatherDay]
    image_urls: List[str]
    source: str  # "vector_store" or "web_search"


# LangGraph State

class AgentState(TypedDict):
    # Conversation history (add_messages merges lists automatically)
    messages: Annotated[list, add_messages]

    # Extracted city name
    city: Optional[str]

    # Routing decision
    use_vector_store: Optional[bool]

    # Intermediate results from parallel nodes
    city_summary: Optional[str]
    weather_forecast: Optional[List[dict]]
    image_urls: Optional[List[str]]

    # Source tracking
    source: Optional[str]

    # Final structured output
    final_response: Optional[dict]

    # Error tracking
    error: Optional[str]
