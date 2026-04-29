# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas",
#     "google-genai",
#     "python-dotenv",
#     "sentence-transformers",
#     "FlagEmbedding",
#     "numpy",
#     "scikit-learn",
#     "plotly",
#     "umap-learn",
#     "pymde",
#     "qdrant-client",
#     "chromadb",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    import json
    import plotly.express as px


    def parse_first_genre(genres_str):
        try:
            genres_list = json.loads(genres_str.replace("'", '"'))
            return genres_list[0]["name"] if genres_list else "Unknown"
        except Exception:
            return "Unknown"


    plot_df = movies_df[["title", "vote_average"]].copy()
    plot_df["genre"] = movies_df["genres"].apply(parse_first_genre)
    plot_df["x"] = mde_coords[:, 0].cpu().numpy()
    plot_df["y"] = mde_coords[:, 1].cpu().numpy()

    # Filter to only show points between -2 and 2
    # plot_df = plot_df[
    #     (plot_df["x"] >= -2)
    #     & (plot_df["x"] <= 2)
    #     & (plot_df["y"] >= -2)
    #     & (plot_df["y"] <= 2)
    # ].reset_index(drop=True)

    embedding_fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="genre",
        hover_name="title",
        hover_data={"vote_average": True, "x": False, "y": False},
        title="PyMDE 2D-Projektion der Film-Embeddings",
        height=600,
    )
    embedding_fig.update_layout(dragmode="select")

    mde_plot = mo.ui.plotly(embedding_fig)
    mde_plot
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Embeddings & Vektordatenbanken

    In diesem Tutorial lernen wir:

    1. **Embeddings erzeugen**, mit drei verschiedenen Modellen (Gemini, Sentence Transformers, BGE-M3)
    2. **Ähnlichkeitssuche**, Kosinus-Ähnlichkeit direkt auf den Vektoren
    3. **Visualisierung**, interaktive 2D-Projektion der Embedding-Vektoren
    4. **Vektordatenbanken**, Qdrant (Cloud) und ChromaDB (lokal)
    5. **Semantische Suche**, Suchfeld mit Live-Ergebnissen aus Qdrant

    Als Datensatz nutzen wir den [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) Datensatz.
    Die **Overview** (Filmbeschreibung) wird vektorisiert, alles andere sind Metadaten.
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from dotenv import load_dotenv

    return Path, load_dotenv, np, os, pd


@app.cell
def _(load_dotenv, os):
    load_dotenv()
    assert os.getenv("GOOGLE_CLOUD_PROJECT"), (
        "GOOGLE_CLOUD_PROJECT nicht in .env gefunden"
    )
    return


@app.cell
def _():
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Verwende Device: {device}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Daten laden
    """)
    return


@app.cell
def _(Path, pd):
    _csv_path = Path("data/tmdb_5000_movies.csv")
    movies_df = pd.read_csv(_csv_path)

    # Nur Filme mit Overview behalten
    movies_df = movies_df.dropna(subset=["overview"])

    # TEST: Nur die 500 am besten bewerteten Filme (mind. 1000 Votes)
    movies_df = (
        movies_df[movies_df["vote_count"] >= 1000]
        .sort_values("vote_average", ascending=False)
        .reset_index(drop=True)
        .head(500)
    )

    print(f"{len(movies_df)} Filme mit Overview geladen")
    movies_df[["title", "overview", "vote_average", "release_date", "genres"]]
    return (movies_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Embeddings erzeugen

    Wir vergleichen drei Embedding-Modelle:

    | Modell | Anbieter | Dimensionen | Besonderheit |
    |--------|----------|-------------|--------------|
    | `gemini-embedding-001` | Google | 3072 | Cloud-API, multilingual |
    | `all-mpnet-base-v2` | Sentence Transformers | 768 | Lokal, schnell, nur Englisch |
    | `BAAI/bge-m3` | FlagEmbedding | 1024 | Lokal, multilingual, multi-granular |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2a) Gemini Embeddings

    Damit Gemini Embeddings richtig über Vertex AI abgerechnet wird, muss man sich einloggen und das `GOOGLE_CLOUD_PROJECT` in der `.env` setzen.

    ```
    gcloud auth application-default login
     ```
    """)
    return


@app.cell
def _(os):
    from google import genai

    gemini_client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
    )
    return (gemini_client,)


@app.cell
def _(gemini_client, movies_df, np):
    _batch_size = 200
    _overviews = movies_df["overview"].tolist()
    gemini_embeddings = []

    _num_batches = (len(_overviews) + _batch_size - 1) // _batch_size
    for _i in range(0, len(_overviews), _batch_size):
        _batch = _overviews[_i : _i + _batch_size]

        _result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=_batch,
        )
        gemini_embeddings.extend([e.values for e in _result.embeddings])

    gemini_embeddings = np.array(gemini_embeddings)
    print(f"\nGemini Embeddings Shape: {gemini_embeddings.shape}")
    return (gemini_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2b) Sentence Transformers, `all-mpnet-base-v2`

    [Sentence Transformers](https://www.sbert.net/) ist eine Python-Bibliothek für lokale Embedding-Modelle.
    `all-mpnet-base-v2` ist eines der besten allgemeinen englischen Modelle.
    """)
    return


@app.cell
def _(device, movies_df, np):
    from sentence_transformers import SentenceTransformer

    _st_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    st_embeddings = _st_model.encode(
        movies_df["overview"].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    st_embeddings = np.array(st_embeddings)
    print(f"Sentence Transformer Embeddings Shape: {st_embeddings.shape}")
    return (st_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2c) BGE-M3

    [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) ist ein multilinguales Embedding-Modell von BAAI.
    Es unterstützt drei Retrieval-Modi: Dense, Sparse (SPLADE-style) und Multi-Vector (ColBERT).
    Wir berechnen Dense und Sparse, Sparse wird später für Hybrid Search in Qdrant genutzt.
    """)
    return


@app.cell
def _(device, movies_df, np):
    from FlagEmbedding import BGEM3FlagModel

    bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
    _output = bge_model.encode(
        movies_df["overview"].tolist(),
        batch_size=64,
        max_length=512,
        return_dense=True,
        return_sparse=True,
    )
    bge_embeddings = np.array(_output["dense_vecs"])
    bge_lexical_weights = _output[
        "lexical_weights"
    ]  # list of dicts {token_id: weight}
    print(f"BGE-M3 Dense Shape: {bge_embeddings.shape}")
    print(
        f"BGE-M3 Sparse: {len(bge_lexical_weights)} Dokumente, Beispiel-Keys: {list(bge_lexical_weights[0].keys())[:5]}"
    )
    return bge_embeddings, bge_lexical_weights, bge_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Mit Embeddings spielen, Kosinus-Ähnlichkeit

    Ohne Datenbank können wir schon ähnliche Filme finden: Wir berechnen die
    Kosinus-Ähnlichkeit eines Films zu allen anderen und sortieren nach Ähnlichkeit.
    """)
    return


@app.cell
def _(mo):
    model_selector = mo.ui.dropdown(
        options=["gemini", "sentence_transformers", "bge_m3"],
        value="gemini",
        label="Embedding-Modell wählen",
    )
    model_selector
    return (model_selector,)


@app.cell
def _(bge_embeddings, gemini_embeddings, model_selector, st_embeddings):
    _embedding_map = {
        "gemini": gemini_embeddings,
        "sentence_transformers": st_embeddings,
        "bge_m3": bge_embeddings,
    }
    selected_embeddings = _embedding_map[model_selector.value]
    print(
        f"Ausgewählt: {model_selector.value}, Shape: {selected_embeddings.shape}"
    )
    return (selected_embeddings,)


@app.cell
def _(mo, movies_df):
    movie_search = mo.ui.dropdown(
        options=movies_df["title"].tolist(),
        value="Avatar",
        label="Film für Ähnlichkeitssuche",
        searchable=True,
    )
    movie_search
    return (movie_search,)


@app.cell
def _(mo, movie_search, movies_df, pd, selected_embeddings):
    # Siehe https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    from sklearn.metrics.pairwise import cosine_similarity

    # Film finden (exakter Titel aus Dropdown)
    _mask = movies_df["title"] == movie_search.value
    if _mask.sum() == 0:
        mo.output.replace(
            mo.md(f"**Kein Film gefunden für:** *{movie_search.value}*")
        )
    else:
        _idx = movies_df[_mask].index[0]
        _query_vec = selected_embeddings[_idx].reshape(1, -1)
        _similarities = cosine_similarity(_query_vec, selected_embeddings)[0]

        _result_df = pd.DataFrame(
            {
                "Titel": movies_df["title"],
                "Ähnlichkeit": _similarities,
                "Overview": movies_df["overview"],
                "Bewertung": movies_df["vote_average"],
                "Genre": movies_df["genres"],
            }
        )
        _result_df = (
            _result_df.sort_values("Ähnlichkeit", ascending=False)
            .iloc[0:11]  # Top 10 ohne den Film selbst
            .reset_index(drop=True)
        )

        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        f"### Top 10 ähnliche Filme zu *{movies_df.loc[_idx, 'title']}*"
                    ),
                    _result_df,
                ]
            )
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Interaktive 2D-Projektion der Embeddings

    Wir projizieren die hochdimensionalen Vektoren mit [PyMDE](https://pymde.org/) auf 2 Dimensionen.
    PyMDE (Minimum-Distortion Embedding) erhält die Nachbarschaftsstruktur besser als UMAP bei
    großen Abständen. **Punkte auswählen** (Box Select oder Lasso) zeigt die zugehörigen Filme als Tabelle.

    Die Visualisierung erzeugt auch eine **Animation** (`play()`), die zeigt, wie sich die Punkte während der
    Optimierung bewegen. Die Punkte sind nach Genre eingefärbt.
    """)
    return


@app.cell
def _(mo):
    @mo.persistent_cache
    def compute_mde_v3(embeddings_key, embeddings, device, genres_str=None):
        import marimo as _mo
        import pymde
        import torch
        import numpy as np
        import json
        from pathlib import Path

        _mo.output.append(
            _mo.md("MDE wird berechnet ... einen Moment bitte.").callout(
                kind="warn"
            )
        )

        data = torch.tensor(embeddings, dtype=torch.float32)
        mde = pymde.preserve_neighbors(
            data,
            embedding_dim=2,
            constraint=pymde.Standardized(),
            device=device,
            verbose=True,
        )

        # Parse genres for coloring
        if genres_str is not None:

            def parse_first_genre(g):
                try:
                    genres_list = json.loads(g.replace("'", '"'))
                    return genres_list[0]["name"] if genres_list else "Unknown"
                except Exception:
                    return "Unknown"

            color_by = np.array([parse_first_genre(g) for g in genres_str])
        else:
            color_by = None

        # Embed with snapshots and create GIF animation
        gif_path = Path(f"mde_animation_{embeddings_key}.gif")
        coords = mde.embed(snapshot_every=3, verbose=True)
        mde.play(
            color_by=color_by,
            axis_limits=(-2, 2),
            marker_size=3.0,
            savepath=str(gif_path),
        )
        _mo.output.clear()
        return coords, str(gif_path)

    return (compute_mde_v3,)


@app.cell
def _(compute_mde_v3, device, model_selector, movies_df, selected_embeddings):
    mde_coords, gif_path = compute_mde_v3(
        model_selector.value,
        selected_embeddings,
        device,
        movies_df["genres"].tolist(),
    )
    return (gif_path,)


@app.cell
def _(gif_path, mo):
    mo.vstack(
        [
            mo.md("### MDE Optimierungs-Animation"),
            mo.md(
                "Die Animation zeigt, wie sich die Punkte während der Optimierung bewegen:"
            ),
            mo.image(src=gif_path),
        ]
    )
    return


@app.cell
def _(mde_plot, mo, movies_df):
    mo.stop(
        not mde_plot.indices,
        mo.md("*Punkte im Diagramm auswählen um Filme zu sehen.*"),
    )
    selected_df = (
        movies_df[["title", "vote_average", "release_date", "genres"]]
        .iloc[mde_plot.indices]
        .copy()
    )
    selected_df.columns = ["Titel", "Bewertung", "Erscheinungsjahr", "Genres"]
    selected_df["Erscheinungsjahr"] = selected_df["Erscheinungsjahr"].str[:4]
    mo.ui.table(selected_df.reset_index(drop=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Vektordatenbanken

    ### 5a) Qdrant (Cloud)

    [Qdrant](https://qdrant.tech/) ist ein Berliner Unternehmen für Vektordatenbanken.
    Wir nutzen einen Cloud-Cluster, URL und API Key kommen aus der `.env`.

    ```
    QDRANT_URL=https://...
    QDRANT_API_KEY=...
    ```
    """)
    return


@app.cell
def _(os):
    from qdrant_client import QdrantClient

    qdrant_client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    print("Qdrant verbunden:", qdrant_client.get_collections())
    return (qdrant_client,)


@app.cell
def _(gemini_embeddings, qdrant_client):
    from qdrant_client.models import (
        Distance,
        Modifier,
        SparseVectorParams,
        VectorParams,
    )

    collection_name = "tmdb_movies"
    _vector_size = gemini_embeddings.shape[1]

    from qdrant_client.models import PayloadSchemaType

    _existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in _existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=_vector_size, distance=Distance.COSINE),
            },
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="release_year",
            field_schema=PayloadSchemaType.INTEGER,
        )
        print(
            f"Collection '{collection_name}' erstellt (dense={_vector_size}d + BM25 sparse)"
        )
    else:
        print(f"Collection '{collection_name}' existiert bereits")
    return Distance, SparseVectorParams, VectorParams, collection_name


@app.cell
def _(collection_name, gemini_embeddings, movies_df, pd, qdrant_client):
    from qdrant_client.models import Document, PointStruct

    _info = qdrant_client.get_collection(collection_name)
    if _info.points_count == 0:
        _batch_size = 100
        for _i in range(0, len(movies_df), _batch_size):
            _end = min(_i + _batch_size, len(movies_df))
            _points = []
            for _idx in range(_i, _end):
                _row = movies_df.iloc[_idx]
                _release = str(_row.get("release_date", "") or "")
                _payload = {
                    "title": str(_row["title"]),
                    "original_title": str(_row["original_title"]),
                    "overview": str(_row["overview"]),
                    "tagline": str(_row.get("tagline", "") or ""),
                    "status": str(_row.get("status", "") or ""),
                    "release_date": _release,
                    "release_year": int(_release[:4])
                    if len(_release) >= 4 and _release[:4].isdigit()
                    else None,
                    "runtime": float(_row["runtime"])
                    if pd.notna(_row["runtime"])
                    else None,
                    "budget": int(_row["budget"])
                    if pd.notna(_row["budget"])
                    else None,
                    "revenue": int(_row["revenue"])
                    if pd.notna(_row["revenue"])
                    else None,
                    "popularity": float(_row["popularity"]),
                    "vote_average": float(_row["vote_average"]),
                    "vote_count": int(_row["vote_count"]),
                    "original_language": str(_row["original_language"]),
                    "homepage": str(_row.get("homepage", "") or ""),
                    "genres": str(_row.get("genres", "") or ""),
                    "keywords": str(_row.get("keywords", "") or ""),
                    "production_companies": str(
                        _row.get("production_companies", "") or ""
                    ),
                    "production_countries": str(
                        _row.get("production_countries", "") or ""
                    ),
                    "spoken_languages": str(
                        _row.get("spoken_languages", "") or ""
                    ),
                    "tmdb_id": int(_row["id"]) if pd.notna(_row["id"]) else None,
                }
                _points.append(
                    PointStruct(
                        id=_idx,
                        vector={
                            "dense": gemini_embeddings[_idx].tolist(),
                        },
                        payload=_payload,
                    )
                )
            qdrant_client.upsert(
                collection_name=collection_name, wait=True, points=_points
            )
        print(f"{len(movies_df)} Filme in Qdrant hochgeladen (dense + BM25)")
    else:
        print(
            f"Collection hat bereits {_info.points_count} Punkte, Upload übersprungen"
        )
    return Document, PointStruct


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Semantische Suche

    Gib einen Suchbegriff ein und finde semantisch ähnliche Filme in der Qdrant-Datenbank.
    Die Suchanfrage wird mit Gemini in einen Vektor umgewandelt und gegen die gespeicherten
    Film-Overviews abgeglichen.
    """)
    return


@app.cell
def _(mo):
    search_input = mo.ui.text(
        placeholder="z.B. 'a hero saving the world from aliens'",
        label="Semantische Suche",
        full_width=True,
    )
    search_input
    return (search_input,)


@app.cell
def _(collection_name, gemini_client, mo, pd, qdrant_client, search_input):
    if search_input.value:
        _result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[search_input.value],
        )
        _query_vector = _result.embeddings[0].values

        _search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=_query_vector,
            using="dense",
            limit=10,
        )

        _rows = []
        for _point in _search_results.points:
            _rows.append(
                {
                    "Score": round(_point.score, 4),
                    "Titel": _point.payload["title"],
                    "Bewertung": _point.payload.get("vote_average", ""),
                    "Jahr": _point.payload.get("release_year", ""),
                    "Overview": _point.payload.get("overview", ""),
                }
            )

        mo.output.replace(
            mo.vstack(
                [
                    mo.md(f"### Suchergebnisse für: *{search_input.value}*"),
                    pd.DataFrame(_rows),
                ]
            )
        )
    else:
        mo.output.replace(
            mo.md("*Suchbegriff eingeben um Ergebnisse zu sehen...*")
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Filtered Search, Semantische Suche mit Jahresfilter

    Finde semantisch ähnliche Filme, die in einem bestimmten Zeitraum erschienen sind.
    """)
    return


@app.cell
def _(mo):
    filter_query = mo.ui.text(
        placeholder="z.B. 'space adventure'",
        label="Suchanfrage",
        value="space adventure",
        full_width=True,
    )
    year_from = mo.ui.slider(
        start=1900,
        stop=2025,
        step=1,
        value=1970,
        label="Jahr von",
        show_value=True,
    )
    year_to = mo.ui.slider(
        start=1900,
        stop=2025,
        step=1,
        value=2015,
        label="Jahr bis",
        show_value=True,
    )
    mo.vstack([filter_query, mo.hstack([year_from, year_to])])
    return filter_query, year_from, year_to


@app.cell
def _(
    collection_name,
    filter_query,
    gemini_client,
    mo,
    pd,
    qdrant_client,
    year_from,
    year_to,
):
    from qdrant_client.models import Filter, FieldCondition, Range

    if filter_query.value:
        _result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[filter_query.value],
        )
        _query_vector = _result.embeddings[0].values

        _search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=_query_vector,
            using="dense",
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="release_year",
                        range=Range(gte=year_from.value, lte=year_to.value),
                    )
                ]
            ),
            limit=10,
        )

        _rows = []
        for _point in _search_results.points:
            _rows.append(
                {
                    "Score": round(_point.score, 4),
                    "Titel": _point.payload["title"],
                    "Jahr": _point.payload.get("release_year", ""),
                    "Bewertung": _point.payload.get("vote_average", ""),
                    "Overview": _point.payload.get("overview", ""),
                }
            )

        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        f"### Filtered: *{filter_query.value}* ({year_from.value}-{year_to.value})"
                    ),
                    pd.DataFrame(_rows),
                ]
            )
        )
    else:
        mo.output.replace(mo.md("*Suchbegriff eingeben...*"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hybrid Search, Reciprocal Rank Fusion (RRF)

    Kombiniert **Dense** (Gemini-Semantik) und **Sparse** (BM25-Keywords) über
    [Reciprocal Rank Fusion](https://qdrant.tech/documentation/concepts/hybrid-queries/).
    Jede Suche liefert eine Rangliste, RRF fusioniert beide zu einem gemeinsamen Ranking.
    """)
    return


@app.cell
def _():
    from qdrant_client.models import Fusion, FusionQuery, Prefetch

    return Fusion, FusionQuery, Prefetch


@app.cell
def _(mo):
    hybrid_query = mo.ui.text(
        placeholder="zB mafia boss is replaced by his son",
        label="Hybrid-Suche (RRF)",
        full_width=True,
    )
    hybrid_query
    return (hybrid_query,)


@app.cell
def _(
    Document,
    Fusion,
    FusionQuery,
    Prefetch,
    gemini_client,
    hybrid_query,
    mo,
    pd,
    qdrant_client,
):
    if hybrid_query.value:
        _result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[hybrid_query.value],
        )
        _query_vector = _result.embeddings[0].values

        _search_results = qdrant_client.query_points(
            collection_name="tmdb_movies",
            prefetch=[
                Prefetch(
                    query=Document(text=hybrid_query.value, model="Qdrant/bm25"),
                    using="sparse",
                    limit=20,
                ),
                Prefetch(
                    query=_query_vector,
                    using="dense",
                    limit=20,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=20,
        )

        _rows = []
        for _point in _search_results.points:
            _rows.append(
                {
                    "Score": round(_point.score, 4),
                    "Titel": _point.payload["title"],
                    "Jahr": _point.payload.get("release_year", ""),
                    "Bewertung": _point.payload.get("vote_average", ""),
                    "Overview": _point.payload.get("overview", ""),
                }
            )

        mo.output.replace(
            mo.vstack(
                [
                    mo.md(f"### Hybrid RRF: *{hybrid_query.value}*"),
                    pd.DataFrame(_rows),
                ]
            )
        )
    else:
        mo.output.replace(mo.md("*Suchbegriff eingeben...*"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Hybrid Search BGE-M3, Dense + SPLADE + RRF

    Gleiche Anfrage wie oben, aber mit BGE-M3 Dense + SPLADE Sparse statt Gemini + BM25.
    BGE-M3 Sparse ist ein **gelerntes** Sparse-Modell, es versteht Synonyme
    ("mafia" vs. "crime family") ohne explizite Query-Expansion.
    """)
    return


@app.cell
def _(
    Distance,
    SparseVectorParams,
    VectorParams,
    bge_embeddings,
    qdrant_client,
):
    bge_collection_name = "tmdb_movies_bge"
    _vector_size = bge_embeddings.shape[1]  # 1024

    _existing = [c.name for c in qdrant_client.get_collections().collections]
    if bge_collection_name not in _existing:
        qdrant_client.create_collection(
            collection_name=bge_collection_name,
            vectors_config={
                "dense": VectorParams(size=_vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                # Kein IDF-Modifier: BGE-M3 Sparse sind bereits gelernte Gewichte (SPLADE-style)
                "sparse": SparseVectorParams(),
            },
        )
        print(
            f"Collection '{bge_collection_name}' erstellt (BGE-M3 dense={_vector_size}d + SPLADE sparse)"
        )
    else:
        print(f"Collection '{bge_collection_name}' existiert bereits")
    return (bge_collection_name,)


@app.cell
def _(
    PointStruct,
    bge_collection_name,
    bge_embeddings,
    bge_lexical_weights,
    movies_df,
    pd,
    qdrant_client,
):
    from qdrant_client.models import SparseVector

    _info = qdrant_client.get_collection(bge_collection_name)

    if _info.points_count == 0:
        _batch_size = 100
        for _i in range(0, len(movies_df), _batch_size):
            _end = min(_i + _batch_size, len(movies_df))
            _points = []
            for _idx in range(_i, _end):
                _row = movies_df.iloc[_idx]
                _release = str(_row.get("release_date", "") or "")
                _lw = bge_lexical_weights[_idx]
                _payload = {
                    "title": str(_row["title"]),
                    "original_title": str(_row["original_title"]),
                    "overview": str(_row["overview"]),
                    "tagline": str(_row.get("tagline", "") or ""),
                    "status": str(_row.get("status", "") or ""),
                    "release_date": _release,
                    "release_year": int(_release[:4])
                    if len(_release) >= 4 and _release[:4].isdigit()
                    else None,
                    "runtime": float(_row["runtime"])
                    if pd.notna(_row["runtime"])
                    else None,
                    "budget": int(_row["budget"])
                    if pd.notna(_row["budget"])
                    else None,
                    "revenue": int(_row["revenue"])
                    if pd.notna(_row["revenue"])
                    else None,
                    "popularity": float(_row["popularity"]),
                    "vote_average": float(_row["vote_average"]),
                    "vote_count": int(_row["vote_count"]),
                    "original_language": str(_row["original_language"]),
                    "genres": str(_row.get("genres", "") or ""),
                    "keywords": str(_row.get("keywords", "") or ""),
                    "tmdb_id": int(_row["id"]) if pd.notna(_row["id"]) else None,
                }
                _points.append(
                    PointStruct(
                        id=_idx,
                        vector={
                            "dense": bge_embeddings[_idx].tolist(),
                            "sparse": SparseVector(
                                indices=[int(k) for k in _lw.keys()],
                                values=[float(v) for v in _lw.values()],
                            ),
                        },
                        payload=_payload,
                    )
                )
            qdrant_client.upsert(
                collection_name=bge_collection_name, wait=True, points=_points
            )
        print(
            f"{len(movies_df)} Filme in '{bge_collection_name}' hochgeladen (BGE-M3 dense + SPLADE)"
        )
    else:
        print(
            f"Collection hat bereits {_info.points_count} Punkte, Upload übersprungen"
        )
    return (SparseVector,)


@app.cell
def _(mo):
    bge_hybrid_query = mo.ui.text(
        placeholder="z.B. 'mafia boss is replaced by his son'",
        label="BGE-M3 Hybrid-Suche (RRF)",
        value="mafia boss is replaced by his son",
        full_width=True,
    )
    bge_hybrid_query
    return (bge_hybrid_query,)


@app.cell
def _(
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
    bge_collection_name,
    bge_hybrid_query,
    bge_model,
    mo,
    pd,
    qdrant_client,
):
    if bge_hybrid_query.value:
        _out = bge_model.encode(
            [bge_hybrid_query.value],
            return_dense=True,
            return_sparse=True,
        )
        _dense_vec = _out["dense_vecs"][0].tolist()
        _lw = _out["lexical_weights"][0]
        _sparse_vec = SparseVector(
            indices=[int(k) for k in _lw.keys()],
            values=[float(v) for v in _lw.values()],
        )

        _results = qdrant_client.query_points(
            collection_name=bge_collection_name,
            prefetch=[
                Prefetch(query=_sparse_vec, using="sparse", limit=20),
                Prefetch(query=_dense_vec, using="dense", limit=20),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=20,
        )

        _rows = []
        for _point in _results.points:
            _rows.append(
                {
                    "Score": round(_point.score, 4),
                    "Titel": _point.payload["title"],
                    "Jahr": _point.payload.get("release_year", ""),
                    "Bewertung": _point.payload.get("vote_average", ""),
                    "Overview": _point.payload.get("overview", ""),
                }
            )

        mo.output.replace(
            mo.vstack(
                [
                    mo.md(f"### BGE-M3 Hybrid RRF: *{bge_hybrid_query.value}*"),
                    pd.DataFrame(_rows),
                ]
            )
        )
    else:
        mo.output.replace(mo.md("*Suchbegriff eingeben...*"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. ChromaDB (lokal)

    [ChromaDB](https://www.trychroma.com/) ist eine Open-Source-Vektordatenbank,
    die lokal auf dem Dateisystem persistiert werden kann. ChromaDB kann eine
    Embedding-Funktion direkt integrieren, sodass beim Einfügen automatisch Vektoren erzeugt werden.

    Wir nutzen hier die Sentence Transformer Embeddings, die wir schon berechnet haben.
    """)
    return


@app.cell
def _(movies_df, pd, st_embeddings):
    import chromadb

    _persist_dir = "./chroma_tmdb"
    chroma_client = chromadb.PersistentClient(path=_persist_dir)

    _collection_name = "tmdb_movies"

    # Collection erstellen oder holen
    chroma_collection = chroma_client.get_or_create_collection(
        name=_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Nur einfügen wenn Collection leer ist
    if chroma_collection.count() == 0:
        _batch_size = 500
        for _i in range(0, len(movies_df), _batch_size):
            _end = min(_i + _batch_size, len(movies_df))
            _ids = [str(_j) for _j in range(_i, _end)]
            _documents = movies_df["overview"].iloc[_i:_end].tolist()
            _embeddings = st_embeddings[_i:_end].tolist()
            _metadatas = [
                {
                    "title": str(row["title"]),
                    "original_title": str(row["original_title"]),
                    "tagline": str(row.get("tagline", "") or ""),
                    "status": str(row.get("status", "") or ""),
                    "release_date": str(row.get("release_date", "") or ""),
                    "runtime": float(row["runtime"])
                    if pd.notna(row["runtime"])
                    else 0.0,
                    "budget": int(row["budget"]) if pd.notna(row["budget"]) else 0,
                    "revenue": int(row["revenue"])
                    if pd.notna(row["revenue"])
                    else 0,
                    "popularity": float(row["popularity"]),
                    "vote_average": float(row["vote_average"]),
                    "vote_count": int(row["vote_count"]),
                    "original_language": str(row["original_language"]),
                    "homepage": str(row.get("homepage", "") or ""),
                    "genres": str(row.get("genres", "") or ""),
                    "keywords": str(row.get("keywords", "") or ""),
                    "production_companies": str(
                        row.get("production_companies", "") or ""
                    ),
                    "production_countries": str(
                        row.get("production_countries", "") or ""
                    ),
                    "spoken_languages": str(row.get("spoken_languages", "") or ""),
                    "tmdb_id": int(row["id"]) if pd.notna(row["id"]) else 0,
                }
                for _, row in movies_df.iloc[_i:_end].iterrows()
            ]
            chroma_collection.add(
                ids=_ids,
                documents=_documents,
                embeddings=_embeddings,
                metadatas=_metadatas,
            )
        print(f"{len(movies_df)} Filme in ChromaDB eingefügt")
    else:
        print(
            f"ChromaDB Collection hat bereits {chroma_collection.count()} Einträge"
        )
    return (chroma_collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **ChromaDB Suche**, Beispiel: "mafia movie"
    """)
    return


@app.cell
def _(chroma_collection):
    chroma_results = chroma_collection.query(
        query_texts=["mafia movie"],
        n_results=5,
    )
    chroma_results
    return


if __name__ == "__main__":
    app.run()
