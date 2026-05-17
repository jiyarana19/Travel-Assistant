import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


CITY_WEATHER_PROFILES = {
    "tokyo": {
        "base_high": 28, "base_low": 21,
        "conditions": ["Partly Cloudy", "Sunny", "Humid", "Light Rain", "Clear"],
    },
    "paris": {
        "base_high": 22, "base_low": 14,
        "conditions": ["Overcast", "Partly Cloudy", "Light Rain", "Clear", "Windy"],
    },
    "new york": {
        "base_high": 26, "base_low": 17,
        "conditions": ["Sunny", "Partly Cloudy", "Thunderstorm", "Clear", "Breezy"],
    },
    "kyoto": {
        "base_high": 30, "base_low": 22,
        "conditions": ["Humid", "Partly Cloudy", "Light Rain", "Clear", "Hot"],
    },
    "default": {
        "base_high": 24, "base_low": 15,
        "conditions": ["Partly Cloudy", "Sunny", "Clear", "Windy", "Overcast"],
    },
}


async def fetch_weather_forecast(city: str) -> List[Dict[str, Any]]:
    await asyncio.sleep(0.5)
    profile = CITY_WEATHER_PROFILES.get(city.lower(), CITY_WEATHER_PROFILES["default"])
    forecast = []
    for i in range(7):
        date = (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d")
        variation = random.uniform(-3, 3)
        high = round(profile["base_high"] + variation, 1)
        low = round(profile["base_low"] + variation - random.uniform(2, 5), 1)
        forecast.append({
            "date": date,
            "temp_high": high,
            "temp_low": low,
            "condition": random.choice(profile["conditions"]),
            "humidity": random.randint(45, 85),
        })
    return forecast


CITY_IMAGE_URLS = {
    "tokyo": [
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=800",
        "https://images.unsplash.com/photo-1513407030348-c983a97b98d8?w=800",
        "https://images.unsplash.com/photo-1524413840807-0c3cb6fa808d?w=800",
        "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?w=800",
    ],
    "paris": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?w=800",
        "https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?w=800",
        "https://images.unsplash.com/photo-1471623432079-b009d30b6729?w=800",
    ],
    "new york": [
        "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?w=800",
        "https://images.unsplash.com/photo-1500916434205-0c77489c6cf7?w=800",
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?w=800",
        "https://images.unsplash.com/photo-1490644658840-3f2e3f8c5625?w=800",
    ],
    "kyoto": [
        "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=800",
        "https://images.unsplash.com/photo-1528360983277-13d401cdc186?w=800",
        "https://images.unsplash.com/photo-1524413840807-0c3cb6fa808d?w=800",
        "https://images.unsplash.com/photo-1545569341-9eb8b30979d9?w=800",
    ],
}


async def fetch_city_images(city: str, count: int = 4) -> List[str]:
    await asyncio.sleep(0.4)

    city_lower = city.lower().strip()

    if city_lower in CITY_IMAGE_URLS:
        return CITY_IMAGE_URLS[city_lower][:count]

    # let it be any city  — Wikipedia API will bring images
    try:
        import urllib.request
        import json as _json
        encoded = city.strip().replace(" ", "_")
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "TravelBot/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = _json.loads(resp.read())
            thumb = data.get("thumbnail", {}).get("source", "")
            original = data.get("originalimage", {}).get("source", "")
            urls = [u for u in [original, thumb] if u]
            # Agar sirf 1-2 images aayi toh seed se baki bharengi
            seed = abs(hash(city)) % 1000
            while len(urls) < count:
                urls.append(f"https://picsum.photos/seed/{seed + len(urls)}/800/600")
            return urls[:count]
    except Exception:
        #  if wikipedia fails then we can use picsum fallback
        seed = abs(hash(city)) % 1000
        return [f"https://picsum.photos/seed/{seed + i}/800/600" for i in range(count)]

async def mock_web_search(query: str) -> str:
    await asyncio.sleep(0.8)
    city = query.replace("travel guide", "").replace("city info", "").strip().title()
    return f"{city} is a fascinating travel destination with rich culture and history. Visitors enjoy its unique architecture, local cuisine, and vibrant street life. The city has grown rapidly while preserving traditional customs. Popular activities include exploring local markets, visiting museums, and experiencing the nightlife. Best time to visit is spring or autumn when weather is pleasant and outdoor events are in full swing."