# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pydantic-ai[google,openai]",
#     "python-dotenv",
#     "qdrant-client",
#     "google-genai",
#     "openai",
#     "numpy",
#     "httpx",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Tool Use & Agenten

    In diesem Notebook bauen wir einen Film-Agenten mit zwei Tools:

    1. **RAG-Suche** (`search_movies_rag`): Semantische Suche in der Qdrant-Filmdatenbank,
       optional gefiltert nach Genre und Erscheinungsjahr
    2. **Beliebte Filme** (`get_popular_movies`): Aktuelle Charts von der TMDB API

    **Framework**: [Pydantic AI](https://ai.pydantic.dev/), ein schlankes, typsicheres
    Agenten-Framework. Tools werden als normale Python-Funktionen definiert und per Decorator
    beim Agenten registriert. Kein Graph, kein Boilerplate.

    **Agentic Loop**: Das LLM entscheidet selbst, welche Tools es aufruft und in welcher
    Reihenfolge. Der Agent laeuft so lange, bis das LLM eine endgueltige Antwort liefert.
    """)
    return


@app.cell
def _():
    import os
    import json
    import httpx
    from dataclasses import dataclass
    from dotenv import load_dotenv
    from google import genai
    from openai import OpenAI
    from qdrant_client import QdrantClient

    load_dotenv()

    gemini_embed_client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
    )

    from qdrant_client.models import TextIndexParams, TokenizerType

    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    qdrant.create_payload_index(
        collection_name="tmdb_movies",
        field_name="genres",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            lowercase=True,
        ),
    )

    tmdb_api_key = os.environ["TMDB_API_KEY"]
    return (
        OpenAI,
        dataclass,
        gemini_embed_client,
        httpx,
        json,
        os,
        qdrant,
        tmdb_api_key,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Modellauswahl
    """)
    return


@app.cell
def _(mo):
    provider_selector = mo.ui.dropdown(
        options=["Gemini", "OpenWebUI"],
        value="Gemini",
        label="Anbieter",
    )
    provider_selector
    return (provider_selector,)


@app.cell
def _(OpenAI, mo, os, provider_selector):
    gemini_models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    owui_client = OpenAI(
        base_url="https://iwschat.service.kitegg.hs-mainz.de/api/",
        api_key=os.getenv("OPEN_WEB_UI_API_KEY"),
    )
    owui_models = [m.id for m in owui_client.models.list().data]

    available_models = (
        gemini_models if provider_selector.value == "Gemini" else owui_models
    )
    model_selector = mo.ui.dropdown(
        options=available_models,
        value=available_models[-2]
        if len(available_models) >= 2
        else available_models[0],
        label="Modell",
    )
    model_selector
    return (model_selector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tools & Agent

    In Pydantic AI werden Tools als Funktionen mit dem `@agent.tool` Decorator definiert.
    Der `RunContext` gibt dem Tool Zugriff auf Abhaengigkeiten (Datenbankclients, API-Keys),
    die beim Aufruf injiziert werden, aehnlich wie Dependency Injection in FastAPI.

    ```python
    @agent.tool
    async def my_tool(ctx: RunContext[MyDeps], param: str) -> str:
        result = await ctx.deps.some_client.query(param)
        return result
    ```
    """)
    return


@app.cell
def _(dataclass, httpx, model_selector, os, provider_selector):
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.google import GoogleProvider
    from qdrant_client.models import Filter, FieldCondition, MatchText, Range
    from typing import Optional


    @dataclass
    class AgentDeps:
        gemini_embed_client: object
        qdrant: object
        tmdb_api_key: str


    if provider_selector.value == "Gemini":
        pai_model = GoogleModel(
            model_selector.value,
            provider=GoogleProvider(
                api_key=os.getenv("GEMINI_API_KEY"),
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location="us-central1",
            ),
        )
    else:
        pai_model = OpenAIModel(
            model_selector.value,
            base_url="https://iwschat.service.kitegg.hs-mainz.de/api/",
            api_key=os.getenv("OPEN_WEB_UI_API_KEY"),
        )

    agent = Agent(
        pai_model,
        deps_type=AgentDeps,
        system_prompt=(
            "You are CineBot, a helpful movie assistant. "
            "Use the available tools to answer questions about movies. "
            "For current trends and charts, use get_popular_movies. "
            "For recommendations based on themes, mood, or descriptions, use search_movies_rag. "
            "For search_movies_rag, write an English plot-focused query describing themes, "
            "atmosphere, and character dynamics -- the query is matched against English movie "
            "plot descriptions, so descriptive English phrases retrieve better results than "
            "simple genre keywords. "
            "Always respond in the same language the user writes in."
        ),
    )


    @agent.tool
    async def search_movies_rag(
        ctx: RunContext[AgentDeps],
        query: str,
        genre: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        k: int = 5,
    ) -> str:
        """Search for movies in the vector database using semantic similarity.

        The query is matched against each movie's English plot description (overview).
        Write a descriptive English query focusing on plot themes, atmosphere, and character
        dynamics rather than genre labels alone. For example, instead of 'mafia movie' use
        'powerful crime family struggles for control as loyalties are tested'.

        Args:
            query: English plot-focused description. Descriptive phrases yield better results
                than single genre keywords.
            genre: Optional genre filter applied directly in the database
                (e.g. 'Action', 'Comedy', 'Drama', 'Crime', 'Thriller').
            year_from: Optional lower bound for release year (inclusive).
            year_to: Optional upper bound for release year (inclusive).
            k: Number of results to return, default 5, max 20.
        """
        embed_result = ctx.deps.gemini_embed_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query],
        )
        query_vec = embed_result.embeddings[0].values

        filter_conditions = []
        if genre:
            filter_conditions.append(
                FieldCondition(key="genres", match=MatchText(text=genre))
            )
        if year_from or year_to:
            filter_conditions.append(
                FieldCondition(
                    key="release_year",
                    range=Range(
                        gte=year_from if year_from else None,
                        lte=year_to if year_to else None,
                    ),
                )
            )

        qdrant_filter = (
            Filter(must=filter_conditions) if filter_conditions else None
        )

        points = ctx.deps.qdrant.query_points(
            collection_name="tmdb_movies",
            query=query_vec,
            using="dense",
            limit=min(k, 50),
            query_filter=qdrant_filter,
        ).points

        if not points:
            return "No movies found matching the given criteria."

        lines = []
        for i, point in enumerate(points, 1):
            p = point.payload
            year = p.get("release_year") or (
                p.get("release_date", "")[:4] if p.get("release_date") else "?"
            )
            lines.append(
                f"{i}. **{p.get('title', 'Unknown')}** ({year})"
                f" - Rating: {p.get('vote_average', 'N/A')}"
                f" - {p.get('overview', '')[:150]}..."
            )

        return "\n".join(lines)


    @agent.tool
    async def get_popular_movies(
        ctx: RunContext[AgentDeps], page: int = 1, language: str = "en-US"
    ) -> str:
        """Fetch the current list of popular movies from the TMDB API.

        Args:
            page: Page number for pagination, default 1. Each page returns up to 20 movies.
            language: BCP 47 language tag for response localization (e.g. 'en-US', 'de-DE',
                'fr-FR'). Set this to match the language the user is writing in.
        """
        url = "https://api.themoviedb.org/3/movie/popular"
        headers = {"accept": "application/json"}
        params = {
            "api_key": ctx.deps.tmdb_api_key,
            "language": language,
            "page": page,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        movies = data.get("results", [])[:10]
        lines = []
        for i, movie in enumerate(movies, 1):
            release_year = movie.get("release_date", "")[:4] or "?"
            lines.append(
                f"{i}. **{movie['title']}** ({release_year})"
                f" - Rating: {movie.get('vote_average', 'N/A'):.1f}"
                f" - {movie.get('overview', '')}"
            )

        return f"Popular movies (page {page}):\n" + "\n".join(lines)

    return AgentDeps, agent


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Chat

    Der Agent ist bereit. Im Chat werden die aufgerufenen Tools und ihre Argumente
    direkt in der Antwort angezeigt, so laesst sich nachvollziehen, wie der Agent "denkt".
    """)
    return


@app.cell
def _(AgentDeps, agent, gemini_embed_client, json, qdrant, tmdb_api_key):
    from pydantic_ai.messages import ToolCallPart

    pai_history = []


    async def chat_fn(messages, config):
        user_msg = messages[-1].content

        result = await agent.run(
            user_msg,
            message_history=pai_history,
            deps=AgentDeps(
                gemini_embed_client=gemini_embed_client,
                qdrant=qdrant,
                tmdb_api_key=tmdb_api_key,
            ),
        )

        pai_history.clear()
        pai_history.extend(result.all_messages())

        tool_lines = []
        for msg in result.new_messages():
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    args_display = (
                        part.args
                        if isinstance(part.args, str)
                        else json.dumps(part.args, ensure_ascii=False)
                    )
                    tool_lines.append(
                        f"🔧 **`{part.tool_name}`** mit `{args_display}`"
                    )

        response_text = result.output
        if tool_lines:
            return "\n".join(tool_lines) + "\n\n---\n\n" + response_text
        return response_text

    return (chat_fn,)


@app.cell
def _(chat_fn, mo):
    chat = mo.ui.chat(
        chat_fn,
        prompts=[
            "Welche Filme sind gerade beliebt?",
            "Empfiehl mir 3 Actionfilme aus den 90ern",
            "Finde Mafia-Dramen zwischen 2000 und 2015",
            "Was sind gute Science-Fiction Filme mit hoher Bewertung?",
            "What are the top rated comedies from the 2000s?",
        ],
        show_configuration_controls=True,
    )
    chat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Zusammenfassung

    | Konzept | Beschreibung |
    |---------|--------------|
    | **Agent** | LLM mit Zugriff auf Tools, das eigenstaendig entscheidet, welche Tools es aufruft |
    | **Tool** | Python-Funktion, die das LLM aufrufen kann, mit Name, Beschreibung und typisiertem Schema |
    | **RunContext** | Dependency Injection fuer Tools: gibt Zugriff auf Clients und API-Keys |
    | **Agentic Loop** | Das LLM ruft Tools auf, verarbeitet die Ergebnisse und wiederholt dies bis zur Endantwort |
    | **message_history** | Gespraechsverlauf, der dem LLM uebergeben wird, damit es den Kontext kennt |
    """)
    return


if __name__ == "__main__":
    app.run()
