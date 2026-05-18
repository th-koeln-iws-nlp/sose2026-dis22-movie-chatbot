# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "google-genai",
#     "openai",
#     "python-dotenv",
#     "FlagEmbedding",
#     "qdrant-client",
#     "numpy",
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
    # Retrieval-Augmented Generation (RAG)

    RAG kombiniert klassische Informationssuche mit generativen Sprachmodellen.
    Aufbauend auf den Notebooks zu Prompt Engineering, Embeddings und Retrieval-Strategien
    bauen wir hier eine vollstandige RAG-Pipeline fur den Film-Chatbot.

    **Ablauf:**

    1. **Retrieve**, Nutzeranfrage encodieren und semantisch ahnliche Filme aus Qdrant abrufen
    2. **Augment**, abgerufene Filme als strukturierten Kontext (XML-Tags) an den Prompt hangen
    3. **Generate**, LLM antwortet auf Basis des Kontexts, ohne zu halluzinieren

    Die Vektordatenbank und das Embedding-Modell werden automatisch auf die gewählte Collection
    abgestimmt: Gemini-Embeddings (3072d) fur `tmdb_movies`, BGE-M3 (1024d) fur die BGE-Collections.
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    from dotenv import load_dotenv
    from google import genai
    from openai import OpenAI

    load_dotenv()

    gen_client = genai.Client(vertexai=True, api_key=os.getenv("GEMINI_API_KEY"))

    gemini_embed_client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
    )

    owui_client = OpenAI(
        base_url="https://iwschat.service.kitegg.hs-mainz.de/api/",
        api_key=os.getenv("OPEN_WEB_UI_API_KEY"),
    )

    gemini_models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    owui_models = [m.id for m in owui_client.models.list().data]
    return (
        gemini_embed_client,
        gemini_models,
        gen_client,
        os,
        owui_client,
        owui_models,
    )


@app.cell
def _(os):
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    available_collections = [c.name for c in qdrant.get_collections().collections]
    print("Verbundene Collections:", available_collections)
    return available_collections, qdrant


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup: Modell und Retrieval-Konfiguration

    Wahle den Anbieter, das Modell und die Temperature fur die Antwortgenerierung.
    Die **Temperature** steuert die Kreativitat der Antworten (0 = deterministisch, 1 = kreativ).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    provider_selector = mo.ui.dropdown(
        options=["Gemini", "OpenWebUI"],
        value="Gemini",
        label="Anbieter",
    )
    provider_selector
    return (provider_selector,)


@app.cell(hide_code=True)
def _(gemini_models, mo, owui_models, provider_selector):
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
    temperature_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.7,
        label="Temperature",
        show_value=True,
    )
    mo.vstack([model_selector, temperature_slider])
    return model_selector, temperature_slider


@app.cell
def _(
    gen_client,
    model_selector,
    owui_client,
    provider_selector,
    temperature_slider,
):
    from google.genai import types


    def complete(user_message=None, system=None, messages=None, temperature=None):
        """Hilfsfunktion: ruft das gewählte LLM auf."""
        temp = temperature if temperature is not None else temperature_slider.value

        if messages is not None:
            all_messages = messages
        else:
            all_messages = []
            if system:
                all_messages.append({"role": "system", "content": system})
            if user_message:
                all_messages.append({"role": "user", "content": user_message})

        if provider_selector.value == "Gemini":
            sys_instruction = next(
                (m["content"] for m in all_messages if m["role"] == "system"), None
            )
            contents = [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in all_messages
                if m["role"] != "system"
            ]
            config_kwargs = {"temperature": temp}
            if sys_instruction:
                config_kwargs["system_instruction"] = sys_instruction
            resp = gen_client.models.generate_content(
                model=model_selector.value,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return resp.text
        else:
            resp = owui_client.chat.completions.create(
                model=model_selector.value,
                messages=all_messages,
                temperature=temp,
            )
            return resp.choices[0].message.content

    return (complete,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Retrieval-Konfiguration

    Wahle die Qdrant-Collection, die Retrieval-Strategie und die Anzahl abzurufender Dokumente.

    | Collection | Embedding | Sparse |
    |---|---|---|
    | `tmdb_movies` | Gemini (3072d) | BM25 |
    | `tmdb_movies_bge` | BGE-M3 (1024d) | SPLADE |
    | `tmdb_movies_bge_full` | BGE-M3 (1024d) | SPLADE + ColBERT |

    **Dense**, nur Vektorsuche (semantisch). **Hybrid**, Dense + Sparse via Reciprocal Rank Fusion (RRF).
    """)
    return


@app.cell
def _(available_collections, mo):
    collection_selector = mo.ui.dropdown(
        options=available_collections,
        value=available_collections[0] if available_collections else None,
        label="Qdrant Collection",
    )
    k_slider = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=5,
        label="Anzahl Dokumente (k)",
        show_value=True,
    )
    strategy_selector = mo.ui.dropdown(
        options=["dense", "hybrid"],
        value="hybrid",
        label="Retrieval-Strategie",
    )
    mo.vstack([collection_selector, strategy_selector, k_slider])
    return collection_selector, k_slider, strategy_selector


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## BGE-M3 Embedding-Modell

    BGE-M3 wird fur Collections mit 1024-dimensionalen Dense-Vektoren verwendet.
    Fur `tmdb_movies` (3072d) wird stattdessen die Gemini Embedding API genutzt.
    Das Modell wird einmalig geladen und gecacht.
    """)
    return


@app.cell
def _(mo):
    @mo.persistent_cache
    def load_bge_model():
        import torch
        from FlagEmbedding import BGEM3FlagModel

        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=dev)
        return model


    bge_model = load_bge_model()
    print("BGE-M3 geladen")
    return (bge_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Prompt-Konfiguration

    Der **System-Prompt** legt das Verhalten des Chatbots fest.
    Die **Nutzeranfrage** wird encodiert, fur die Vektorsuche genutzt und
    dann zusammen mit dem abgerufenen Kontext an das LLM weitergegeben.
    """)
    return


@app.cell
def _(mo):
    system_prompt_input = mo.ui.text_area(
        value=(
            "Du bist CineBot, ein hilfreicher Film-Chatbot.\n"
            "Du empfiehlst Filme basierend auf den Nutzerpraferenzen.\n"
            "Nutze ausschliesslich die Filme aus dem bereitgestellten Kontext.\n"
            "Erfinde keine Filme, die nicht im Kontext stehen.\n"
            "Antworte auf Deutsch, freundlich und pragnant."
        ),
        label="System-Prompt",
        full_width=True,
        rows=6,
    )
    user_query_input = mo.ui.text_area(
        value="Filme über Mafia Familien",
        label="Nutzeranfrage",
        full_width=True,
        rows=3,
    )
    mo.vstack([system_prompt_input, user_query_input])
    return system_prompt_input, user_query_input


@app.cell
def _(
    bge_model,
    collection_selector,
    gemini_embed_client,
    qdrant,
    strategy_selector,
    user_query_input,
):
    from qdrant_client.models import SparseVector

    col_info = qdrant.get_collection(collection_selector.value)
    vectors_config = col_info.config.params.vectors

    if isinstance(vectors_config, dict):
        dense_size = vectors_config["dense"].size
    else:
        dense_size = vectors_config.size

    if dense_size == 3072:
        detected_encoder = "Gemini"
        embed_result = gemini_embed_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[user_query_input.value],
        )
        query_dense_vec = embed_result.embeddings[0].values
    else:
        detected_encoder = "BGE-M3"
        query_output = bge_model.encode(
            [user_query_input.value],
            return_dense=True,
            return_sparse=True,
        )
        query_dense_vec = query_output["dense_vecs"][0].tolist()

    query_sparse_vec = None
    if strategy_selector.value == "hybrid" and detected_encoder == "BGE-M3":
        sparse_weights = bge_model.encode(
            [user_query_input.value],
            return_dense=False,
            return_sparse=True,
        )["lexical_weights"][0]
        query_sparse_vec = SparseVector(
            indices=[int(k) for k in sparse_weights.keys()],
            values=[float(v) for v in sparse_weights.values()],
        )

    print(f"Encoder: {detected_encoder}, Dense-Dimension: {dense_size}")
    return col_info, detected_encoder, query_dense_vec, query_sparse_vec


@app.cell
def _(
    col_info,
    collection_selector,
    detected_encoder,
    k_slider,
    qdrant,
    query_dense_vec,
    query_sparse_vec,
    strategy_selector,
    user_query_input,
):
    from qdrant_client.models import Document, Fusion, FusionQuery, Prefetch

    sparse_vector_names = (
        list(col_info.config.params.sparse_vectors.keys())
        if col_info.config.params.sparse_vectors
        else []
    )

    if strategy_selector.value == "dense" or not sparse_vector_names:
        retrieved_points = qdrant.query_points(
            collection_name=collection_selector.value,
            query=query_dense_vec,
            using="dense",
            limit=k_slider.value,
        ).points
    else:
        if detected_encoder == "Gemini":
            sparse_name = "bm25_sparse_vector"
            sparse_query = Document(
                text=user_query_input.value, model="Qdrant/bm25"
            )
        else:
            sparse_name = "sparse"
            sparse_query = query_sparse_vec

        retrieved_points = qdrant.query_points(
            collection_name=collection_selector.value,
            prefetch=[
                Prefetch(
                    query=sparse_query, using=sparse_name, limit=k_slider.value * 2
                ),
                Prefetch(
                    query=query_dense_vec, using="dense", limit=k_slider.value * 2
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k_slider.value,
        ).points

    print(
        f"Abgerufen: {len(retrieved_points)} Dokumente via {strategy_selector.value}"
    )
    return (retrieved_points,)


@app.cell
def _(retrieved_points):
    retrieved_points
    return


@app.cell
def _(retrieved_points, user_query_input):
    film_blocks = []
    for film_idx, point in enumerate(retrieved_points, start=1):
        p = point.payload
        block = (
            f'<film id="{film_idx}">\n'
            f"Titel: {p.get('title', '')}\n"
            f"Overview: {p.get('overview', '')}\n"
            f"Bewertung: {p.get('vote_average', '')}\n"
            f"Jahr: {p.get('release_year', p.get('release_date', '')[:4] if p.get('release_date') else '')}\n"
            f"Genre: {p.get('genres', '')}\n"
            f"</film>"
        )
        film_blocks.append(block)

    kontext_xml = "<kontext>\n" + "\n".join(film_blocks) + "\n</kontext>"

    assembled_prompt = f"## USER PROMPT\n{user_query_input.value}\n\n{kontext_xml}"
    return (assembled_prompt,)


@app.cell
def _(assembled_prompt, mo):
    mo.vstack(
        [
            mo.md("### Zusammengestellter Prompt mit Kontext"),
            mo.md(
                "Der folgende Prompt wird an das LLM gesendet. Der Kontext ist mit XML-Tags strukturiert:"
            ),
            mo.md(f"```\n{assembled_prompt}\n```"),
        ]
    )
    return


@app.cell
def _(assembled_prompt, complete, system_prompt_input):
    llm_response = complete(
        user_message=assembled_prompt,
        system=system_prompt_input.value,
    )
    return (llm_response,)


@app.cell(hide_code=True)
def _(llm_response, mo, retrieved_points):
    import pandas as pd

    result_rows = [
        {
            "Rang": i + 1,
            "Score": round(p.score, 4),
            "Titel": p.payload.get("title", ""),
            "Jahr": p.payload.get("release_year", ""),
            "Bewertung": p.payload.get("vote_average", ""),
        }
        for i, p in enumerate(retrieved_points)
    ]

    mo.vstack(
        [
            mo.md("### Antwort von CineBot"),
            mo.md(llm_response),
            mo.md("---"),
            mo.md("**Abgerufene Dokumente aus Qdrant:**"),
            mo.ui.table(pd.DataFrame(result_rows), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Zusammenfassung

    | Schritt | Beschreibung |
    |---|---|
    | **Retrieve** | Nutzeranfrage wird encodiert, Qdrant sucht semantisch ahnliche Filme |
    | **Augment** | Gefundene Filme werden als XML-Kontext an den Prompt angehangt |
    | **Generate** | LLM generiert Antwort, beschrankt auf den bereitgestellten Kontext |

    ### Wichtige Beobachtungen

    - **Halluzinationen reduzieren:** Durch den System-Prompt ("Erfinde keine Filme...") und den expliziten
      Kontext wird das Modell auf die tatsachlich vorhandenen Daten beschrankt.
    - **Retrieval-Qualitat entscheidet:** Relevante Ergebnisse setzen gute Embeddings voraus.
      Hybrid-Suche ist robuster als reine Dense-Suche.
    - **XML-Tags:** Strukturieren den Kontext klar und helfen dem Modell,
      Anweisung und Daten zu trennen (siehe Notebook Prompt Engineering).
    - **k-Parameter:** Zu wenige Dokumente = fehlender Kontext. Zu viele = Context-Window-Uberlauf
      und sinkende Antwortqualitat.
    """)
    return


if __name__ == "__main__":
    app.run()
