import os
import time
from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("Neo4j_URI", "neo4j+s://demo.neo4jlabs.com"),
    username=os.getenv("Neo4j_USER", "recommendations"),
    password=os.getenv("Neo4j_PASSWORD", "recommendations"),
    database=os.getenv("Neo4j_DATABASE", "recommendations"),
    enhanced_schema=True,
)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def get_movies_by_plot_similarity(question: str, k: int = 10) -> list[dict[str, Any]]:
    """
    Retrieve information about movies based on vector similarity search. This will retrieve similar movies based on the plot description.

    Parameters
    ----------
    question: str
        The question to answer.
    k: int
        The number of movies to retrieve.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing the movie information.
        Each dictionary contains the movile title, actors that starred in the movie, the movie's IMDB rating, and the movie's plot.
    """

    question_embedding = embedder.embed_query(question)
    query = """
    CALL db.index.vector.queryNodes('moviePlotsEmbedding', $k, $question_embedding)
    YIELD node as movie
    MATCH (actor)-[:ACTED_IN]->(movie:Movie)
    RETURN  COLLECT(actor.name) as actors, 
            movie.title as title, 
            movie.imdbRating as rating,
            movie.plot as plot
    ORDER BY rating DESC
    LIMIT $k
    """
    params={"question_embedding": question_embedding, "k": k}
    return graph.query(query, params=params)

@tool
def get_user_favorite_genres_by_name(user_name: str) -> list[dict[str, Any]]:
    """
    Retrieve information about the user's favorite genres. This will retrieve the user's favorite genres and their average rating.
    
    Parameters
    ----------
    user_name: str
        The name of the user to retrieve the favorite genres for.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing the user's favorite genres.
        Each dictionary contains the genre name and the average rating for that genre.
    """
    query = """
MATCH (u:User {name: $user_name})-[:RATED]->(m:Movie)-[:IN_GENRE]->(g:Genre)
WHERE m.imdbRating IS NOT NULL
WITH g.name as genre, AVG(m.imdbRating) as avg_rating
ORDER BY avg_rating DESC
RETURN genre, ROUND(avg_rating, 2) as average_rating
LIMIT 10
    """
    params={"user_name": user_name}
    return graph.query(query, params=params)

agent = create_react_agent(
    model=llm,
    tools=[get_movies_by_plot_similarity, get_user_favorite_genres_by_name],
)


# __________________________________________________________
# Streamlit app
# __________________________________________________________    

st.title("Movie Explorer Agent")

def chat_stream(prompt):
    response = agent.stream({"messages": [("user", prompt)]})
    for chunk in response:
        if chunk.get("agent"):
            yield chunk["agent"]["messages"][-1].content


def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]


if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(chat_stream(prompt))
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )
    st.session_state.history.append({"role": "assistant", "content": response})