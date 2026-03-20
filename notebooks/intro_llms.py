# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas",
#     "google-genai",
#     "openai",
#     "python-dotenv",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import pandas as pd
    from pathlib import Path
    from dotenv import load_dotenv
    from google import genai

    return Path, genai, load_dotenv, os, pd


@app.cell
def _(genai, load_dotenv, os):
    load_dotenv()
    _api_key = os.getenv("GEMINI_API_KEY")
    # Plain client for listing models (vertexai=True requires OAuth2 for list endpoint)
    list_client = genai.Client(api_key=_api_key)
    # Vertex client for content generation
    gen_client = genai.Client(vertexai=True, api_key=_api_key)
    return gen_client, list_client


@app.cell
def _(Path):
    data_path = Path(__file__).parent.parent / "data" / "imdb_top_1000.csv"
    return (data_path,)


@app.cell
def _(data_path, pd):
    df = pd.read_csv(data_path)
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.table(df)
    return


@app.cell
def _(list_client):
    # Filter out non-chat models (embeddings, audio, TTS, image generation, etc.)
    _excluded = {
        "embedding",
        "audio",
        "tts",
        "robotics",
        "image",
        "computer-use",
        "native",
    }
    available_models = [
        m.name.replace("models/", "")
        for m in list_client.models.list()
        if "gemini" in m.name.lower()
        and not any(x in m.name.lower() for x in _excluded)
    ]
    return (available_models,)


@app.cell
def _(available_models, mo):
    _default = (
        "gemini-2.5-flash"
        if "gemini-2.5-flash" in available_models
        else available_models[0]
    )
    model_dropdown = mo.ui.dropdown(
        options=available_models,
        value=_default,
        label="Model",
    )
    mo.vstack(
        [
            mo.md("## Chat"),
            model_dropdown,
        ]
    )
    return (model_dropdown,)


@app.cell
def _(gen_client, mo, model_dropdown):
    def call_gemini(messages, config):
        # Convert marimo ChatMessage list to Gemini contents format
        # Gemini uses "model" for the assistant role
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
        ]
        resp = gen_client.models.generate_content(
            model=model_dropdown.value,
            contents=contents,
        )
        return resp.text


    chat = mo.ui.chat(
        call_gemini,
        prompts=["Recommend a movie", "What's a good thriller from the 90s?"],
        show_configuration_controls=True,
    )
    chat
    return


@app.cell
def _(os):
    from openai import OpenAI

    owui_client = OpenAI(
        base_url="https://iwschat.service.kitegg.hs-mainz.de/api/",
        api_key=os.getenv("OPEN_WEB_UI_API_KEY"),
    )
    owui_models = [m.id for m in owui_client.models.list().data]
    return owui_client, owui_models


@app.cell
def _(mo, owui_models):
    owui_dropdown = mo.ui.dropdown(
        options=owui_models,
        value=owui_models[0] if owui_models else None,
        label="Model",
    )
    mo.vstack(
        [
            mo.md("## OpenWebUI Chat"),
            owui_dropdown,
        ]
    )
    return (owui_dropdown,)


@app.cell
def _(mo, owui_client, owui_dropdown):
    def call_owui(messages, config):
        response = owui_client.chat.completions.create(
            model=owui_dropdown.value,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return response.choices[0].message.content


    owui_chat = mo.ui.chat(
        call_owui,
        prompts=["Recommend a movie", "What's a good thriller from the 90s?"],
        show_configuration_controls=True,
    )
    owui_chat
    return


if __name__ == "__main__":
    app.run()
