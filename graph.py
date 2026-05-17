"""
graph.py — LangGraph topology assembly.

Architecture:
    [extract_city] → [router] → {vector_store | web_search} → [parallel_fetch] → [aggregator]

Distinction 2 (Parallel Fan-Out): parallel_fetch_node runs weather + image APIs concurrently.
Distinction 3 (Memory/Checkpointer): MemorySaver preserves conversation context across turns.

Run `python graph.py` to export graph.png for your README.
"""

import asyncio
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import (
    extract_city_node,
    router_node,
    vector_store_node,
    web_search_node,
    parallel_fetch_node,
    aggregator_node,
    route_by_knowledge,
)


def build_graph(use_checkpointer: bool = True):
    """
    Construct and compile the LangGraph state machine.

    Args:
        use_checkpointer: If True, enables memory (Distinction 3).
                         Pass thread_id in config to preserve context across turns.

    Returns:
        Compiled LangGraph app.
    """
    builder = StateGraph(AgentState)

    # ── Register Nodes ─────────────────────────────────────────────────────────
    builder.add_node("extract_city", extract_city_node)
    builder.add_node("router", router_node)
    builder.add_node("vector_store", vector_store_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("parallel_fetch", parallel_fetch_node)
    builder.add_node("aggregator", aggregator_node)

    # ── Entry Point ────────────────────────────────────────────────────────────
    builder.set_entry_point("extract_city")

    # ── Fixed Edges ────────────────────────────────────────────────────────────
    builder.add_edge("extract_city", "router")

    # ── Conditional Edge: The "Switch" ─────────────────────────────────────────
    # Routes to vector_store if city is known, web_search otherwise
    builder.add_conditional_edges(
        "router",
        route_by_knowledge,
        {
            "vector_store": "vector_store",
            "web_search": "web_search",
        },
    )

    # ── Both paths converge at parallel_fetch ──────────────────────────────────
    builder.add_edge("vector_store", "parallel_fetch")
    builder.add_edge("web_search", "parallel_fetch")

    # ── Parallel fetch → aggregator → END ─────────────────────────────────────
    builder.add_edge("parallel_fetch", "aggregator")
    builder.add_edge("aggregator", END)

    # ── Compile with optional Memory Checkpointer (Distinction 3) ─────────────
    if use_checkpointer:
        checkpointer = MemorySaver()
        app = builder.compile(checkpointer=checkpointer)
    else:
        app = builder.compile()

    return app


async def run_query(city_query: str, thread_id: str = "default") -> dict:
    """
    Run a single travel query through the graph.

    Args:
        city_query: Natural language query, e.g. "Tell me about Kyoto"
        thread_id: Session identifier for memory (Distinction 3).
                   Same thread_id = same conversation context.

    Returns:
        Final structured response dict.
    """
    from langchain_core.messages import HumanMessage

    app = build_graph(use_checkpointer=True)

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=city_query)],
        "city": None,
        "use_vector_store": None,
        "city_summary": None,
        "weather_forecast": None,
        "image_urls": None,
        "source": None,
        "final_response": None,
        "error": None,
    }

    result = await app.ainvoke(initial_state, config=config)
    return result.get("final_response", {})


async def run_followup(followup_query: str, thread_id: str = "default") -> dict:
    """
    Handle follow-up queries using preserved memory (Distinction 3).

    Example:
        await run_query("Tell me about Tokyo", thread_id="user_123")
        await run_followup("What about next week's weather?", thread_id="user_123")
        # The agent knows city=Tokyo from prior context, only re-fetches weather.

    Args:
        followup_query: Follow-up message leveraging prior context.
        thread_id: Must match the prior run's thread_id.
    """
    from langchain_core.messages import HumanMessage

    app = build_graph(use_checkpointer=True)
    config = {"configurable": {"thread_id": thread_id}}

    # With the help of MemorySaver, checkpointer will restore full prior state automatically
    # We only need to pass the new message; city, summary and all are already persisted
    followup_state = {
        "messages": [HumanMessage(content=followup_query)],
    }

    result = await app.ainvoke(followup_state, config=config)
    return result.get("final_response", {})




def export_graph_image(output_path: str = "graph.png"):
    """Generate a visual topology diagram of the graph."""
    try:
        app = build_graph(use_checkpointer=False)
        png_bytes = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph exported to {output_path}")
    except Exception as e:
        print(f"Could not export graph image: {e}")
        print("Install graphviz: pip install graphviz")


if __name__ == "__main__":
    # test here
    async def _test():
        print("Testing Paris (vector store path)...")
        result = await run_query("Tell me about Paris", thread_id="test_thread")
        print(f"City: {result.get('city')}")
        print(f"Source: {result.get('source')}")
        print(f"Summary length: {len(result.get('city_summary', ''))} chars")
        print(f"Weather days: {len(result.get('weather_forecast', []))}")
        print(f"Images: {len(result.get('image_urls', []))}")

        print("\nTesting Snohomish (web search path)...")
        result2 = await run_query("What about Snohomish?", thread_id="test_thread_2")
        print(f"City: {result2.get('city')}")
        print(f"Source: {result2.get('source')}")

    asyncio.run(_test())
    export_graph_image()
