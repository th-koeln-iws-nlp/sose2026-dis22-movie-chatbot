# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas",
#     "python-dotenv",
#     "FlagEmbedding",
#     "numpy",
#     "qdrant-client",
#     "plotly",
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
    # Retrieval Strategien

    Aufbauend auf Sitzung 02 (Embeddings & Vektordatenbanken) vergleichen wir heute vier Retrieval-Strategien in Qdrant:

    1. **Sparse (BM25)**, exakte Keyword-Suche
    2. **Dense**, semantische Vektorsuche mit Bi-Encoder
    3. **Hybrid (RRF)**, Dense und Sparse fusioniert via Reciprocal Rank Fusion
    4. **Multi-Vector (ColBERT)**, token-genaues Late-Interaction-Matching mit BGE-M3

    Als Datensatz nutzen wir wieder den [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) Datensatz
    mit den 500 am besten bewerteten Filmen.
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
    assert os.getenv("QDRANT_URL"), "QDRANT_URL nicht in .env gefunden"
    assert os.getenv("QDRANT_API_KEY"), "QDRANT_API_KEY nicht in .env gefunden"
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
    csv_path = Path("data/tmdb_5000_movies.csv")
    movies_df = pd.read_csv(csv_path)
    movies_df = movies_df.dropna(subset=["overview"])
    movies_df = (
        movies_df[movies_df["vote_count"] >= 1000]
        .sort_values("vote_average", ascending=False)
        .head(500)
        .reset_index(drop=True)
    )
    print(f"{len(movies_df)} Filme geladen")
    movies_df[["title", "overview", "vote_average"]].head(5)
    return (movies_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Drei Retrieval-Paradigmen

    ### 2a) Sparse Retrieval: BM25

    BM25 berechnet einen Score pro Dokument basierend auf **Wortübereinstimmungen**, ohne ein Embedding-Modell.
    Drei Komponenten bestimmen den Score:

    | Komponente | Funktion |
    |---|---|
    | **Term Frequency (TF)** | Wie oft kommt der Suchbegriff im Dokument vor? Mit Sattigung: ab einem Punkt bringen weitere Vorkommen weniger. |
    | **Inverse Document Frequency (IDF)** | Wie selten ist der Begriff uber alle Dokumente? "the" hat niedrigen IDF, "Interstellar" hohen. |
    | **Langennormalisierung** | Langere Dokumente haben mehr Terme. BM25 normalisiert dafur. |

    **Starke:** Exakte Keyword-Matches, Titel, Namen, Jahreszahlen.
    **Schwache:** "Gangster" findet keine Dokumente mit "Mafiosi", Synonyme und Umschreibungen werden verpasst.

    Neuere **Sparse-Vektoren** (SPLADE, BGE-M3 Sparse) kombinieren das Prinzip mit neuronalen Netzen:
    Sie erzeugen gewichtete Token-Vektoren und konnen auch verwandte Terme aktivieren, die nicht im Text stehen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2b) Dense Retrieval: Bi-Encoder

    Query und Dokument werden **unabhangig** durch ein Encoder-Modell geleitet, je ein dichter Vektor pro Text.
    Die Ahnlichkeit wird per Kosinus-Ahnlichkeit berechnet. Dokumente konnen vorab berechnet werden,
    nur die Query wird zur Laufzeit eingebettet.

    **Starke:** Synonyme, Paraphrasen, konzeptuelle Nahe. "Film uber Einsamkeit" findet *Interstellar* und *Gravity*.
    **Schwache:** Ein ganzer Text wird auf einen einzigen Vektor komprimiert. Exakte Terme, Jahreszahlen und
    Eigennamen konnen dabei verloren gehen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2c) Multi-Vector: ColBERT (Late Interaction)

    ColBERT ist ein Mittelweg zwischen schnellem Bi-Encoder und prazisem Cross-Encoder:

    1. Query und Dokument werden **getrennt** encodiert, aber alle Token-Vektoren werden behalten (nicht nur einer).
    2. Fur jedes Query-Token wird der **maximale** Ahnlichkeitsscore uber alle Dokument-Token berechnet (MaxSim).
    3. Diese MaxSim-Scores werden aufsummiert zum finalen Relevanz-Score.

    $$\text{score}(Q, D) = \sum_{i} \max_{j} \, E_q[i] \cdot E_d[j]^\top$$

    **Starke:** Token "Interstellar" in der Query matcht direkt mit "Interstellar" im Dokument.
    Token "einsam" matcht mit "isolated" oder "alone". Feingranularer als Bi-Encoder, schneller als Cross-Encoder,
    da Dokument-Token vorab berechnet werden konnen.

    **Schwache:** Deutlich mehr Speicherbedarf als Single-Vector (ein Vektor pro Token statt pro Dokument).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. BGE-M3: Alle drei Modi in einem Modell

    [BGE-M3](https://arxiv.org/abs/2402.03216) (BAAI, 2024) liefert in einem einzigen Encoding-Durchlauf drei Outputs:

    | Modus | Output | Qdrant-Vektortyp |
    |---|---|---|
    | **Dense** | CLS-Vektor (1024d) | `VectorParams` |
    | **Sparse** | Token-Gewichte (SPLADE-Style) | `SparseVectorParams` |
    | **Multi-Vector** | Alle Token-Vektoren (n_tokens x 1024d) | `VectorParams` + `MultiVectorConfig(MAX_SIM)` |

    568 Millionen Parameter, 100+ Sprachen, Apache 2.0 Lizenz.
    """)
    return


@app.cell
def _(device):
    from FlagEmbedding import BGEM3FlagModel

    bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
    print("BGE-M3 geladen")
    return (bge_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Embeddings berechnen (Dense + Sparse + ColBERT)

    Wir berechnen alle drei Embeddings in einem Durchlauf mit `return_dense=True`,
    `return_sparse=True` und `return_colbert_vecs=True`.
    """)
    return


@app.cell
def _(mo, np):
    def compute_bge_full(overviews_tuple):
        from FlagEmbedding import BGEM3FlagModel
        import torch

        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

        mo.output.append(
            mo.md(
                "BGE-M3 Embeddings (Dense + Sparse + ColBERT) werden berechnet..."
            ).callout(kind="warn")
        )

        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=dev)
        output = model.encode(
            list(overviews_tuple),
            batch_size=32,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

        dense = np.array(output["dense_vecs"])
        sparse = output["lexical_weights"]
        colbert = output["colbert_vecs"]

        mo.output.clear()
        return dense, sparse, colbert

    return (compute_bge_full,)


@app.cell
def _(compute_bge_full, movies_df):
    dense_vecs, sparse_weights, colbert_vecs = compute_bge_full(
        tuple(movies_df["overview"].tolist())
    )
    print(f"Dense Shape:   {dense_vecs.shape}")
    print(f"Sparse:        {len(sparse_weights)} Dokumente")
    print(
        f"ColBERT:       {len(colbert_vecs)} Dokumente, Beispiel-Shape: {colbert_vecs[0].shape}"
    )
    return colbert_vecs, dense_vecs, sparse_weights


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Qdrant Collection einrichten

    Wir erstellen die Collection `tmdb_movies_bge_full` mit allen drei Vektortypen in einem einzigen Schema:

    ```python
    vectors_config={
        "dense":   VectorParams(size=1024, distance=Distance.COSINE),
        "colbert": VectorParams(size=1024, distance=Distance.COSINE,
                       multivector_config=MultiVectorConfig(MultiVectorComparator.MAX_SIM)),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),   # SPLADE-Style, kein IDF-Modifier
    }
    ```
    """)
    return


@app.cell
def _(os):
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    print("Qdrant verbunden:", qdrant.get_collections())
    return (qdrant,)


@app.cell
def _(dense_vecs, qdrant):
    from qdrant_client.models import (
        Distance,
        MultiVectorComparator,
        MultiVectorConfig,
        SparseVectorParams,
        VectorParams,
    )

    collection_name = "tmdb_movies_bge_full"
    vector_size = dense_vecs.shape[1]

    existing_collections = [c.name for c in qdrant.get_collections().collections]
    if collection_name not in existing_collections:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
                "colbert": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        print(
            f"Collection '{collection_name}' erstellt (dense + sparse + colbert, je 1024d)"
        )
    else:
        print(f"Collection '{collection_name}' existiert bereits")
    return (collection_name,)


@app.cell
def _(
    colbert_vecs,
    collection_name,
    dense_vecs,
    movies_df,
    pd,
    qdrant,
    sparse_weights,
):
    from qdrant_client.models import PointStruct, SparseVector

    collection_info = qdrant.get_collection(collection_name)

    if collection_info.points_count == 0:
        batch_size = 15
        for batch_start in range(0, len(movies_df), batch_size):
            batch_end = min(batch_start + batch_size, len(movies_df))
            points = []
            for idx in range(batch_start, batch_end):
                _row = movies_df.iloc[idx]
                release = str(_row.get("release_date", "") or "")
                lw = sparse_weights[idx]

                # Parse first genre only to reduce payload size
                import json

                try:
                    genres_list = json.loads(
                        str(_row.get("genres", "[]")).replace("'", '"')
                    )
                    first_genre = (
                        genres_list[0]["name"] if genres_list else "Unknown"
                    )
                except Exception:
                    first_genre = "Unknown"

                payload = {
                    "title": str(_row["title"]),
                    "overview": str(_row["overview"]),
                    "release_date": release,
                    "release_year": int(release[:4])
                    if len(release) >= 4 and release[:4].isdigit()
                    else None,
                    "vote_average": float(_row["vote_average"]),
                    "vote_count": int(_row["vote_count"]),
                    "genre": first_genre,  # Only first genre, not full JSON string
                    "runtime": float(_row["runtime"])
                    if pd.notna(_row["runtime"])
                    else None,
                }
                points.append(
                    PointStruct(
                        id=idx,
                        vector={
                            "dense": dense_vecs[idx].tolist(),
                            "colbert": colbert_vecs[idx].tolist(),
                            "sparse": SparseVector(
                                indices=[int(k) for k in lw.keys()],
                                values=[float(v) for v in lw.values()],
                            ),
                        },
                        payload=payload,
                    )
                )
            qdrant.upsert(
                collection_name=collection_name, wait=True, points=points
            )
        print(f"{len(movies_df)} Filme hochgeladen (dense + sparse + colbert)")
    else:
        print(
            f"Collection hat bereits {collection_info.points_count} Punkte, Upload ubersprungen"
        )
    return (SparseVector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Retrieval Strategien im Vergleich

    Gib eine Suchanfrage ein. Alle vier Strategien werden gleichzeitig ausgefuhrt und
    die Ergebnisse direkt nebeneinander angezeigt.

    Probier verschiedene Query-Typen:
    - **Exakter Titel:** "The Dark Knight 2008"
    - **Semantisch:** "Film uber Einsamkeit im Weltraum"
    - **Eigenname:** "Christopher Nolan Zeitreise"
    - **Inhaltlich:** "mafia boss replaced by his son"
    """)
    return


@app.cell
def _(mo):
    query_input = mo.ui.text(
        placeholder="z.B. 'The Dark Knight 2008' oder 'Film uber Einsamkeit im Weltraum'",
        label="Suchanfrage",
        value="mafia boss replaced by his son",
        full_width=True,
    )
    query_input
    return (query_input,)


@app.cell
def _(bge_model, query_input):
    query_output = bge_model.encode(
        [query_input.value],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    query_dense_vec = query_output["dense_vecs"][0].tolist()
    query_sparse_lw = query_output["lexical_weights"][0]
    query_colbert_vec = query_output["colbert_vecs"][0].tolist()
    print(f"Query encodiert: {len(query_colbert_vec)} Token-Vektoren fur ColBERT")
    return query_colbert_vec, query_dense_vec, query_sparse_lw


@app.cell
def _(
    SparseVector,
    collection_name,
    qdrant,
    query_colbert_vec,
    query_dense_vec,
    query_sparse_lw,
):
    from qdrant_client.models import Fusion, FusionQuery, Prefetch

    sparse_query_vec = SparseVector(
        indices=[int(k) for k in query_sparse_lw.keys()],
        values=[float(v) for v in query_sparse_lw.values()],
    )

    results_dense = qdrant.query_points(
        collection_name=collection_name,
        query=query_dense_vec,
        using="dense",
        limit=10,
    ).points

    results_sparse = qdrant.query_points(
        collection_name=collection_name,
        query=sparse_query_vec,
        using="sparse",
        limit=10,
    ).points

    results_hybrid = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=sparse_query_vec, using="sparse", limit=20),
            Prefetch(query=query_dense_vec, using="dense", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=10,
    ).points

    results_colbert = qdrant.query_points(
        collection_name=collection_name,
        query=query_colbert_vec,
        using="colbert",
        limit=10,
    ).points
    return results_colbert, results_dense, results_hybrid, results_sparse


@app.cell
def _(
    mo,
    pd,
    query_input,
    results_colbert,
    results_dense,
    results_hybrid,
    results_sparse,
):
    def to_df(points, score_label="Score"):
        return pd.DataFrame(
            [
                {
                    "Rang": i + 1,
                    "Titel": p.payload["title"],
                    score_label: round(p.score, 4),
                }
                for i, p in enumerate(points)
            ]
        )


    df_dense = to_df(results_dense, "Kosinus")
    df_sparse = to_df(results_sparse, "SPLADE")
    df_hybrid = to_df(results_hybrid, "RRF")
    df_colbert = to_df(results_colbert, "MaxSim")

    mo.vstack(
        [
            mo.md(f"### Ergebnisse fur: *{query_input.value}*"),
            mo.hstack(
                [
                    mo.vstack(
                        [mo.md("**Dense**"), mo.ui.table(df_dense, selection=None)]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Sparse (SPLADE)**"),
                            mo.ui.table(df_sparse, selection=None),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Hybrid (RRF)**"),
                            mo.ui.table(df_hybrid, selection=None),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**ColBERT (MaxSim)**"),
                            mo.ui.table(df_colbert, selection=None),
                        ]
                    ),
                ],
                gap=2,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Ranking-Vergleich uber alle Strategien

    Die folgende Visualisierung zeigt, wie sich die Reihenfolge der Top-10-Filme
    je nach Retrieval-Strategie unterscheidet. Ein Film der in allen Strategien weit oben steht,
    ist robust uber verschiedene Query-Typen.
    """)
    return


@app.cell
def _(
    pd,
    query_input,
    results_colbert,
    results_dense,
    results_hybrid,
    results_sparse,
):
    import plotly.express as px
    import plotly.graph_objects as go

    strategies = {
        "Dense": results_dense,
        "Sparse": results_sparse,
        "Hybrid (RRF)": results_hybrid,
        "ColBERT": results_colbert,
    }

    all_titles = []
    for strategy_results in strategies.values():
        for point in strategy_results:
            title = point.payload["title"]
            if title not in all_titles:
                all_titles.append(title)

    rank_rows = []
    for title in all_titles:
        _row = {"Film": title}
        for strategy_name, strategy_results in strategies.items():
            titles_in_strategy = [p.payload["title"] for p in strategy_results]
            _row[strategy_name] = (
                titles_in_strategy.index(title) + 1
                if title in titles_in_strategy
                else None
            )
        rank_rows.append(_row)

    rank_df = (
        pd.DataFrame(rank_rows).sort_values("Hybrid (RRF)").reset_index(drop=True)
    )

    strategy_colors = {
        "Dense": "#636EFA",
        "Sparse": "#EF553B",
        "Hybrid (RRF)": "#00CC96",
        "ColBERT": "#AB63FA",
    }

    rank_fig = go.Figure()
    for strategy_col, color in strategy_colors.items():
        rank_fig.add_trace(
            go.Bar(
                name=strategy_col,
                x=rank_df["Film"],
                y=rank_df[strategy_col],
                marker_color=color,
                opacity=0.8,
            )
        )

    rank_fig.update_layout(
        barmode="group",
        title=f"Rang pro Strategie fur: '{query_input.value}'",
        xaxis_title="Film",
        yaxis_title="Rang (niedriger = besser)",
        yaxis_autorange="reversed",
        legend_title="Strategie",
        height=500,
        xaxis_tickangle=-35,
    )
    rank_fig
    return


if __name__ == "__main__":
    app.run()
