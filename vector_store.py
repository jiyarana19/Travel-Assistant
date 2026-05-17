import chromadb
from chromadb.utils import embedding_functions
from typing import Tuple

CITY_FACTS = {
    "paris": """
    Paris, the capital of France, is known as the City of Light. Home to 2.1 million people,
    Paris sits along the Seine River in northern France. Key attractions include the Eiffel Tower
    built in 1889, the Louvre Museum housing the Mona Lisa, Notre-Dame Cathedral, the Arc de
    Triomphe, and the Musee d Orsay. The Champs-Elysees is one of the world's most famous
    boulevards. Paris is a global center for art, fashion, gastronomy, and culture. The city is
    divided into 20 arrondissements. The Marais district is known for medieval architecture.
    Montmartre is famous for the Sacre-Coeur Basilica. Best time to visit is April to June or
    September to October. Paris has more Michelin-starred restaurants than any other city.
    Transportation includes one of Europe's best metro systems with 16 lines.
    """,
    "tokyo": """
    Tokyo is the capital of Japan with a population of 37.4 million, making it the world's most
    populous metropolitan area. Key districts include Shinjuku, Shibuya famous for its scramble
    crossing, Harajuku, Asakusa with traditional temples, Akihabara for electronics and anime,
    and Ginza for luxury shopping. Tokyo Skytree at 634 meters is the tallest tower in Japan.
    Tokyo has more Michelin-starred restaurants than any city in the world. Best time to visit
    is March to May for cherry blossoms or October to November for autumn foliage. Tokyo's rail
    network includes the Shinkansen bullet train. The city is a global leader in anime, manga,
    gaming, fashion, and technology with over 200 festivals annually.
    """,
    "new york": """
    New York City is the most populous city in the United States with 8.3 million residents.
    It consists of five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island.
    Manhattan is home to Wall Street, Times Square, Central Park with 843 acres, the Empire
    State Building, the Statue of Liberty, the Metropolitan Museum of Art, and the High Line.
    NYC is a global center of finance, media, culture, and fashion. It houses the United Nations
    headquarters. Broadway is the pinnacle of American theater. Best time to visit is April to
    June and September to November. NYC's subway operates 24 hours a day with 472 stations.
    The city is renowned for pizza, bagels, cheesecake, and diverse international cuisine with
    over 26000 restaurants.
    """,
}

ALIASES = {
    "nyc": "new york",
    "new york city": "new york",
    "ny": "new york",
}

_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.EphemeralClient()
    ef = embedding_functions.DefaultEmbeddingFunction()

    _collection = client.get_or_create_collection(
        name="cities",
        embedding_function=ef,
    )

    if _collection.count() == 0:
        _collection.add(
            documents=list(CITY_FACTS.values()),
            ids=list(CITY_FACTS.keys()),
            metadatas=[{"city": k} for k in CITY_FACTS.keys()],
        )

    return _collection


def query_vector_store(city: str) -> Tuple[bool, str]:
    normalized = city.lower().strip()

    # Direct match (fastest path)
    if normalized in CITY_FACTS:
        return True, CITY_FACTS[normalized]

    # Alias match
    if normalized in ALIASES:
        return True, CITY_FACTS[ALIASES[normalized]]

    # Partial match
    for key in CITY_FACTS:
        if key in normalized or normalized in key:
            return True, CITY_FACTS[key]

    # ChromaDB semantic search as final fallback
    try:
        collection = _get_collection()
        results = collection.query(
            query_texts=[city],
            n_results=1,
        )
        if results["distances"] and results["distances"][0]:
            distance = results["distances"][0][0]
            if distance < 0.3:
                doc = results["documents"][0][0]
                return True, doc
    except Exception:
        pass

    return False, ""