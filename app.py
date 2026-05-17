import traceback
import asyncio
import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional
import uuid

st.set_page_config(
    page_title="Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Here is basically all the styling part
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        opacity: 0.8;
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
    }

    .source-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }

    .source-vector {
        background: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }

    .source-web {
        background: #e3f2fd;
        color: #1565c0;
        border: 1px solid #90caf9;
    }

    .city-summary {
        background: #1e1e2e;
        color: #e0e0e0;
        border-left: 4px solid #0f3460;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        line-height: 1.8;
        font-size: 1.02rem;
        margin-bottom: 1.5rem;
    }

    .stImage img {
        border-radius: 10px;
    }

    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }

    .chat-user {
        background: #2a2a3e;
        color: #c9d1e0;
        border-left: 3px solid #5c6bc0;
    }

    .chat-assistant {
        background: #1a2e1a;
        color: #c8e6c9;
        border-left: 3px solid #43a047;
    }

    div[data-testid="stSpinner"] {
        margin: 2rem auto;
    }
</style>
""", unsafe_allow_html=True)


# This is the session State 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "is_followup" not in st.session_state:
    st.session_state.is_followup = False


def run_agent(query: str, is_followup: bool = False) -> Optional[dict]:
    try:
        from graph import run_query, run_followup

        # memory- fixing: last city namee will be used in the followup 
        if is_followup and st.session_state.last_response:
            last_city = st.session_state.last_response.get("city", "")
            if last_city and last_city.lower() != "unknown":
                result = asyncio.run(
                    run_query(
                        f"Tell me about {last_city}",
                        thread_id=st.session_state.thread_id
                    )
                )
                return result

        result = asyncio.run(
            run_query(query, thread_id=st.session_state.thread_id)
        )
        return result
    except Exception as e:
        st.error(traceback.format_exc())
        return None


def render_weather_chart(weather_data: list):
    if not weather_data:
        st.warning("Weather data unavailable.")
        return

    df = pd.DataFrame(weather_data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["temp_high"],
        mode="lines+markers",
        name="High °C",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=8),
        fill="tonexty",
        fillcolor="rgba(231, 76, 60, 0.08)",
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["temp_low"],
        mode="lines+markers",
        name="Low °C",
        line=dict(color="#3498db", width=3),
        marker=dict(size=8),
    ))

    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["humidity"],
        name="Humidity %",
        marker_color="rgba(52, 152, 219, 0.25)",
        yaxis="y2",
    ))

    fig.update_layout(
        title=dict(
            text="7-Day Weather Forecast",
            font=dict(size=18, family="Playfair Display, serif"),
        ),
        xaxis=dict(title="Date", tickangle=-30),
        yaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis2=dict(
            title="Humidity (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 150],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Source Sans Pro, sans-serif"),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_image_gallery(image_urls: list, city: str):
    if not image_urls:
        st.info("No images available for this city.")
        return

    cols = st.columns(min(len(image_urls), 2))
    for i, url in enumerate(image_urls[:4]):
        with cols[i % 2]:
            try:
                st.image(url, caption=f"{city} — Photo {i+1}", use_container_width=True)
            except Exception:
                st.markdown(f"[View Image {i+1}]({url})")


def render_response(response: dict):
    if not response:
        st.error("No response received from the agent.")
        return

    city = response.get("city", "Unknown")
    summary = response.get("city_summary", "")
    weather = response.get("weather_forecast", [])
    images = response.get("image_urls", [])
    source = response.get("source", "unknown")

    badge_class = "source-vector" if source == "vector_store" else "source-web"
    badge_text = "📚 Local Knowledge Base" if source == "vector_store" else "🌐 Live Web Search"
    st.markdown(
        f'<span class="source-badge {badge_class}">{badge_text}</span>',
        unsafe_allow_html=True,
    )

    st.markdown(f"## {city}")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📖 City Overview")
        import re
        clean_summary = re.sub(r'\*+', '', summary)
        st.markdown(f'<div class="city-summary">{clean_summary}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🌤️ Weather Forecast")
        render_weather_chart(weather)

        if weather:
            today = weather[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("High", f"{today['temp_high']}°C")
            m2.metric("Low", f"{today['temp_low']}°C")
            m3.metric("Humidity", f"{today['humidity']}%")
            st.caption(f"Today's condition: **{today['condition']}**")

    st.markdown("---")
    st.markdown("### 📸 Photo Gallery")
    render_image_gallery(images, city)


# code for slidebar
with st.sidebar:
    st.markdown("## ✈️ Travel Assistant")
    st.markdown("---")

    st.markdown("**Pre-loaded cities** (vector store):")
    st.markdown("- 🗼 Paris\n- 🗾 Tokyo\n- 🗽 New York")

    st.markdown("**Other cities** route to web search.")
    st.markdown("---")

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.session_state.is_followup = False
        st.rerun()

    st.markdown("---")
    st.markdown("**Session ID**")
    st.caption(st.session_state.thread_id[:8] + "...")

    st.markdown("---")
    st.markdown("**Architecture**")
    st.caption("LangGraph · ChromaDB · Async Fan-Out · MemorySaver")


# code for Main header 
st.markdown("""
<div class="main-header">
    <h1>✈️ Multi-Modal Travel Assistant</h1>
    <p>Powered by LangGraph · Ask about any city in the world</p>
</div>
""", unsafe_allow_html=True)


# stire chat historyy
if st.session_state.chat_history:
    with st.expander("💬 Conversation History", expanded=False):
        for msg in st.session_state.chat_history:
            role_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="chat-message {role_class}">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )


# takes input
st.markdown("### 🔍 Ask about a city")

placeholder = "e.g. Tell me about Kyoto, What's Paris like? or What about Tokyo?"
if st.session_state.last_response:
    placeholder = "Follow up: 'What about next week?' or ask about another city..."

col_input, col_btn = st.columns([5, 1])
with col_input:
    user_query = st.text_input(
        "City query",
        placeholder=placeholder,
        label_visibility="collapsed",
        key="query_input",
    )
with col_btn:
    search_clicked = st.button("Search", type="primary", use_container_width=True)


# Processes Query in this section
if search_clicked and user_query.strip():
    followup_keywords = ["next week", "how about", "what about", "tomorrow", "forecast", "instead", "and"]
    is_followup = (
        st.session_state.last_response is not None
        and any(kw in user_query.lower() for kw in followup_keywords)
    )

    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner(f"{'Updating context' if is_followup else 'Exploring'} ..."):
        response = run_agent(user_query, is_followup=is_followup)

    if response:
        st.session_state.last_response = response
        city = response.get("city", "Unknown")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Here's a complete travel guide for {city}! Source: {response.get('source', 'N/A')}",
        })
        render_response(response)
    else:
        st.error("Something went wrong. Please check your API key and try again.")
        st.info("Make sure GROQ_API_KEY or ANTHROPIC_API_KEY is set in your environment.")

elif st.session_state.last_response and not search_clicked:
    render_response(st.session_state.last_response)

else:
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### 📚 Smart Routing")
        st.markdown("Known cities use a local vector store. Unknown cities trigger live web search automatically.")
    with cols[1]:
        st.markdown("### ⚡ Parallel Fetching")
        st.markdown("Weather and images are fetched concurrently, cutting response time in half.")
    with cols[2]:
        st.markdown("### 🧠 Memory")
        st.markdown("Follow-up questions like 'What about next week?' preserve city context automatically.")