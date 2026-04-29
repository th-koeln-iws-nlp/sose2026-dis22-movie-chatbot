# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "google-genai",
#     "openai",
#     "pydantic",
#     "instructor",
#     "python-dotenv",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import json
    from dotenv import load_dotenv
    from google import genai
    from openai import OpenAI

    load_dotenv()

    _api_key = os.getenv("GEMINI_API_KEY")
    owui_api_key = os.getenv("OPEN_WEB_UI_API_KEY")

    gen_client = genai.Client(vertexai=True, api_key=_api_key)

    # Gemini-Modelle (hardcoded, listing braucht OAuth ohne vertexai=True)
    gemini_models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    # OpenWebUI-Modelle dynamisch laden
    owui_client = OpenAI(
        base_url="https://iwschat.service.kitegg.hs-mainz.de/api/",
        api_key=owui_api_key,
    )
    owui_models = [m.id for m in owui_client.models.list().data]
    return gemini_models, gen_client, json, owui_client, owui_models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prompt Engineering für Film- & Kino-Chatbots

    Dieses Notebook zeigt grundlegende Prompt-Engineering-Techniken
    anhand von Film- und Kino-Beispielen.

    ## Quellen

    - [Anthropic Prompt Engineering Tutorial](https://github.com/anthropics/courses/tree/master/prompt_engineering_interactive_tutorial)
    - [Lilian Weng's Prompt Engineering Guide](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
    - [Eugene Yan's Prompting Fundamentals](https://eugeneyan.com/writing/prompting/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: Modellauswahl

    Auswahl von Anbieter und Modell für alle Beispiele in diesem Notebook.
    Die **Temperature** steuert die Zufälligkeit der Antworten (0 = deterministisch, 1 = kreativ/zufällig).
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
def _(gemini_models, mo, owui_models, provider_selector):
    _models = gemini_models if provider_selector.value == "Gemini" else owui_models
    model_selector = mo.ui.dropdown(
        options=_models,
        value=_models[-2] if len(_models) >= 2 else _models[0],
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


    def complete(
        user_message=None, system=None, messages=None, temperature=None, **kwargs
    ):
        """Hilfsfunktion: ruft das gewählte LLM auf.

        Entweder `user_message` (+ optional `system`) übergeben,
        oder eine fertige `messages`-Liste.
        `response_format={"type": "json_object"}` wird für beide Anbieter unterstützt.
        """
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
            if kwargs.get("response_format", {}).get("type") == "json_object":
                config_kwargs["response_mime_type"] = "application/json"
            resp = gen_client.models.generate_content(
                model=model_selector.value,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return resp.text
        else:
            create_kwargs = {"temperature": temp}
            if "response_format" in kwargs:
                create_kwargs["response_format"] = kwargs["response_format"]
            resp = owui_client.chat.completions.create(
                model=model_selector.value,
                messages=all_messages,
                **create_kwargs,
            )
            return resp.choices[0].message.content

    return (complete,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 1. Klar und Direkt

    Das grundlegendste Prinzip: **Präzision**. Vage Prompts führen zu vagen Antworten.

    ### Prinzipien
    - Genaue Angabe des Gewünschten, ohne das Modell raten zu lassen
    - Angabe von **Anzahl, Format und Länge** der gewünschten Ausgabe
    - Verwendung von **konkreten Formulierungen** statt allgemeiner Bitten
    - **Bullet oder nummerierte Listen** für mehrere Anforderungen sind hilfreich

    Vergleich zwischen vagem und präzisem Prompt:
    """)
    return


@app.cell
def _(mo):
    vague_input = mo.ui.text_area(
        value="Empfiehl mir Filme.",
        label="Vager Prompt",
        full_width=True,
        rows=2,
    )
    specific_input = mo.ui.text_area(
        value="Empfiehl mir genau 3 Actionfilme aus den 1990er Jahren. Für jeden Film: Titel, Erscheinungsjahr, eine Begründung in maximal 2 Sätzen. Sortiere nach deiner Bewertung, besten Film zuerst.",
        label="Präziser Prompt",
        full_width=True,
        rows=4,
    )
    mo.vstack([vague_input, specific_input])
    return specific_input, vague_input


@app.cell
def _(complete, specific_input, vague_input):
    vague_result = complete(vague_input.value)
    specific_result = complete(specific_input.value)
    return specific_result, vague_result


@app.cell
def _(mo, specific_result, vague_result):
    mo.vstack(
        [
            mo.md(f"### ❌ Vager Prompt\n\n{vague_result}"),
            mo.md("---"),
            mo.md(f"### ✅ Präziser Prompt\n\n{specific_result}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 2. System, User und Assistent Rollen

    Moderne LLMs nutzen eine Gesprächsstruktur mit drei Rollen:

    | Rolle | Bedeutung |
    |-------|-----------|
    | **System** | Definiert Verhalten, Persönlichkeit und Regeln des Modells |
    | **User** | Die Person, die Fragen stellt oder Aufgaben gibt |
    | **Assistent** | Die Antwort des Modells |

    Der **System-Prompt** ist wie eine Stellenbeschreibung:
    Er legt fest, wer das Modell „ist" und welche Regeln es befolgt.
    Allgemeine Aufgaben gehören in den System-Prompt, konkrete Eingaben in die User-Nachricht.
    """)
    return


@app.cell
def _(mo):
    # VORSICHT! Das LLM hat noch gar keine Zugriff auf die Datenbank und wird die Antwort halluzinieren
    system_input = mo.ui.text_area(
        value='Du bist ein Filmempfehlungs-Chatbot namens "CineBot".\nDu kennst ausschließlich die 1000 bestbewerteten Filme auf IMDB.\nAntworte immer auf Deutsch, freundlich und prägnant.\nWenn ein Film nicht in den IMDB Top 1000 ist, weise höflich darauf hin.',
        label="System-Prompt",
        full_width=True,
        rows=5,
    )

    user_input = mo.ui.text_area(
        value="Was sind die besten Mafiafilme? Und gibt es 'Fast & Furious' in deiner Datenbank?",
        label="User-Nachricht",
        full_width=True,
        rows=2,
    )
    mo.vstack([system_input, user_input])
    return system_input, user_input


@app.cell
def _(complete, system_input, user_input):
    role_result = complete(
        user_message=user_input.value, system=system_input.value
    )
    return (role_result,)


@app.cell
def _(mo, role_result):
    mo.md(f"""
    ### Antwort\n\n{role_result}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 3. Daten von Anweisungen trennen

    Wenn Anweisungen und Daten vermischt werden, kann das Modell verwirrt werden,
    besonders wenn die Daten selbst textuelle Anweisungen enthalten (Prompt Injection).

    ## XML-Tags

    **XML-Tags** schaffen klare, unzweideutige Grenzen:

    ```xml
    <anweisung>Was soll das Modell tun?</anweisung>

    <daten>Die zu verarbeitenden Inhalte</daten>
    ```

    Gängige Patterns:
    - `<text>`, `<dokument>`, `<eingabe>` – für Daten
    - `<anweisung>`, `<regeln>` – für Anweisungen
    - `<beispiel>`, `<beispiele>` – für Demonstrations-Beispiele
    - `<kontext>`, `<hintergrund>` – für Zusatzinformationen
    """)
    return


@app.cell
def _(mo):
    review_input = mo.ui.text_area(
        value='Nolan hat es wieder getan. "The Dark Knight" ist nicht einfach ein Superheldenfilm – es ist ein Krimi, ein psychologisches Kammerspiel, eine Studie über Chaos und Moral. Heath Ledgers Joker ist dabei so fesselnd und verstörend wie kaum eine Filmfigur zuvor. Allerdings wirkt das letzte Drittel etwas überladen, und die Entwicklung rund um Harvey Dent fühlt sich gehetzt an. Trotzdem: Pflichtprogramm.',
        label="Filmrezension",
        full_width=True,
        rows=5,
    )
    review_input
    return (review_input,)


@app.cell
def _(complete, review_input):
    xml_prompt = f"""Analysiere die folgende Filmrezension und extrahiere:
    1. Gesamtbewertung: positiv / gemischt / negativ
    2. Genannte Stärken (Stichpunkte)
    3. Genannte Schwächen (Stichpunkte)
    4. Erwähnte Personen (Regisseur, Schauspieler)

    <rezension>
    {review_input.value}
    </rezension>

    Antworte strukturiert mit den vier Punkten."""

    xml_result = complete(xml_prompt)
    return xml_prompt, xml_result


@app.cell
def _(mo, xml_prompt, xml_result):
    mo.vstack(
        [
            mo.md("**Prompt mit XML-Tags:**"),
            mo.md(f"```\n{xml_prompt}\n```"),
            mo.md(f"**Ergebnis:**\n\n{xml_result}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 4. Few-Shot Learning

    Beim **Few-Shot Prompting** zeigt man dem Modell Beispiele der gewünschten Ausgabe,
    bevor man die eigentliche Aufgabe stellt.

    Das ist besonders effizient für:
    - Durchsetzung eines bestimmten **Stils oder Formats**
    - **Konsistente Ausgaben** über viele Anfragen
    - Aufgaben, die schwer zu beschreiben, aber leicht zu zeigen sind

    Vergleich: **Zero-Shot** (keine Beispiele) vs. **Few-Shot** (mit Beispielen im System-Prompt)
    """)
    return


@app.cell
def _(mo):
    film_to_classify = mo.ui.text_area(
        value="Ein ehemaliger Polizist reist in eine verschneite Kleinstadt, um den Mord an einem lokalen Unternehmer aufzuklären, und entdeckt dabei ein Netz aus Korruption und alten Geheimnissen.",
        label="Filmbeschreibung klassifizieren",
        full_width=True,
        rows=3,
    )
    film_to_classify
    return (film_to_classify,)


@app.cell
def _(complete, film_to_classify):
    # Zero-Shot: keine Beispiele
    zero_shot_result = complete(
        f"Welches Genre hat dieser Film?\n\n{film_to_classify.value}"
    )

    # Few-Shot: drei Beispiele im System-Prompt
    few_shot_system = """Du klassifizierst Filmbeschreibungen in Genres.
    Antworte immer im Format: HAUPTGENRE (Nebengenre, Nebengenre) – ein Satz Begründung.

    <beispiele>
    <beispiel>
    Beschreibung: "Zwei Gangster transportieren eine Aktentasche für ihren Boss und philosophieren über Burger und das Leben."
    Genre: CRIME (Schwarze Komödie, Drama) – Tarantinos nicht-lineares Erzählformat und moralisch ambivalente Protagonisten sind typisch für Neo-Noir.
    </beispiel>
    <beispiel>
    Beschreibung: "Eine Gruppe ungleicher Helden muss einen mächtigen Bösewicht aufhalten, bevor dieser eine magische Waffe aktiviert."
    Genre: ACTION (Fantasy, Abenteuer) – Die klassische Heldenreise mit übernatürlichen Elementen und einem klar definierten Antagonisten verortet den Film im Fantasy-Action-Genre.
    </beispiel>
    <beispiel>
    Beschreibung: "Ein Astronaut strandet allein auf dem Mars und muss mit begrenzten Ressourcen überleben, während die NASA eine Rettungsmission plant."
    Genre: SCI-FI (Survival, Drama) – Die wissenschaftlich fundierte Problemlösung und die Isolation des Protagonisten sind Kernmerkmale des Hard-Science-Fiction-Genres.
    </beispiel>
    </beispiele>"""

    few_shot_result = complete(
        user_message=f'Beschreibung: "{film_to_classify.value}"',
        system=few_shot_system,
    )
    return few_shot_result, zero_shot_result


@app.cell
def _(few_shot_result, mo, zero_shot_result):
    mo.vstack(
        [
            mo.md(f"### ❌ Zero-Shot\n\n{zero_shot_result}"),
            mo.md("---"),
            mo.md(f"### ✅ Few-Shot\n\n{few_shot_result}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 5. Chain of Thought (CoT)

    Beim **Chain of Thought** wird der Denkprozess des Modells explizit sichtbar gemacht.
    Das verbessert die Qualität bei komplexen Schlussfolgerungen deutlich.

    - **Zero-Shot CoT**: Erweiterung des Prompts um „Denke Schritt für Schritt"
    - **Standard CoT**: Vorausschicken eines Beispiels mit explizitem Denkprozess

    > **Hinweis:** Viele aktuelle Modelle (Gemini 2.5 Pro, o3, etc.) sind Reasoning-Modelle
    > und führen CoT intern durch – ohne dass man es explizit anfordern muss.
    """)
    return


@app.cell
def _(mo):
    preference_input = mo.ui.text_area(
        value="Ich habe Goodfellas, The Godfather und Scarface geliebt. Ich mag Spannung, komplexe Charaktere und realistische Darstellungen von Kriminalität. Was empfiehlst du mir als Nächstes?",
        label="Filmpräferenz des Nutzers",
        full_width=True,
        rows=3,
    )
    preference_input
    return (preference_input,)


@app.cell
def _(complete, preference_input):
    # Ohne CoT: direkte Antwort
    direct_result = complete(preference_input.value)

    # Mit Zero-Shot CoT
    cot_result = complete(
        preference_input.value
        + "\n\nDenke Schritt für Schritt: Welche konkreten Merkmale haben die genannten Filme gemeinsam? Welche Filme erfüllen exakt diese Kriterien?"
    )
    return cot_result, direct_result


@app.cell
def _(cot_result, direct_result, mo):
    mo.vstack(
        [
            mo.md(f"### Direkte Antwort (ohne CoT)\n\n{direct_result}"),
            mo.md("---"),
            mo.md(f"### Mit Chain of Thought\n\n{cot_result}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 6. Strukturierte Ausgaben

    Häufig werden keine Fließtextantworten benötigt, sondern **Daten**
    zur programmatischen Weiterverarbeitung.

    ## 6.1 JSON Mode

    Moderne LLMs können direkt valides JSON ausgeben.
    """)
    return


@app.cell
def _(mo):
    film_desc_input = mo.ui.text_area(
        value="The Shawshank Redemption ist ein 1994 erschienener amerikanischer Gefängnisfilm von Frank Darabont, basierend auf einer Novelle von Stephen King. Tim Robbins spielt einen zu Unrecht verurteilten Banker, Morgan Freeman seinen weisen Mitgefangenen. Der Film gilt heute als einer der besten Filme aller Zeiten und hat eine IMDB-Bewertung von 9,3.",
        label="Filmbeschreibung",
        full_width=True,
        rows=4,
    )
    film_desc_input
    return (film_desc_input,)


@app.cell
def _(complete, film_desc_input, json):
    _json_prompt = f"""Extrahiere aus der folgenden Filmbeschreibung ein JSON-Objekt mit diesen Feldern:
    - titel (string)
    - jahr (integer)
    - regisseur (string)
    - hauptdarsteller (Liste von strings)
    - genre (Liste von strings)
    - kurzbeschreibung (string, max. 2 Sätze)
    - imdb_bewertung (float)

    <beschreibung>
    {film_desc_input.value}
    </beschreibung>

    Antworte NUR mit dem JSON-Objekt, ohne Erklärungen oder Markdown."""

    json_raw = complete(_json_prompt, response_format={"type": "json_object"})
    json_parsed = json.loads(json_raw)
    return (json_parsed,)


@app.cell
def _(json_parsed, mo):
    mo.json(json_parsed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.2 Instructor: Typsichere strukturierte Ausgaben

    **JSON Mode** hat Einschränkungen: keine Schema-Validierung, keine Typprüfung,
    keine automatischen Wiederholungen bei Fehlern.

    [Instructor](https://github.com/jxnl/instructor) löst das:
    - Typsichere **Pydantic-Modelle**
    - Automatische **Validierung** der Ausgabe
    - Automatische **Wiederholung** bei ungültiger Ausgabe

    Referenz: [Instructor + Google Gemini](https://python.useinstructor.com/integrations/google/)
    """)
    return


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import List


    class FilmDaten(BaseModel):
        """Strukturierte Filmdaten aus einer Beschreibung"""

        titel: str = Field(description="Offizieller Filmtitel")
        jahr: int = Field(description="Erscheinungsjahr", ge=2010, le=2030)
        regisseur: str = Field(description="Name des Regisseurs")
        hauptdarsteller: List[str] = Field(description="Liste der Hauptdarsteller")
        genre: List[str] = Field(description="Filmgenres")
        kurzbeschreibung: str = Field(description="Kurzbeschreibung, max. 2 Sätze")
        imdb_bewertung: float = Field(
            description="IMDB-Bewertung oder Schätzung (1–10)",
            ge=1.0,
            le=10.0,
        )

    return (FilmDaten,)


@app.cell
def _(
    FilmDaten,
    film_desc_input,
    gen_client,
    model_selector,
    owui_client,
    provider_selector,
    temperature_slider,
):
    from google.genai import types as _types

    if provider_selector.value == "Gemini":
        _resp = gen_client.models.generate_content(
            model=model_selector.value,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"Extrahiere die Filmdaten:\n\n{film_desc_input.value}"
                        }
                    ],
                }
            ],
            config=_types.GenerateContentConfig(
                system_instruction="Du extrahierst strukturierte Filmdaten aus Beschreibungen.",
                response_mime_type="application/json",
                response_schema=FilmDaten,
                temperature=temperature_slider.value,
            ),
        )
        instructor_result = FilmDaten.model_validate_json(_resp.text)
    else:
        import instructor

        _client = instructor.from_openai(owui_client)
        instructor_result = _client.chat.completions.create(
            model=model_selector.value,
            response_model=FilmDaten,
            messages=[
                {
                    "role": "system",
                    "content": "Du extrahierst strukturierte Filmdaten aus Beschreibungen.",
                },
                {
                    "role": "user",
                    "content": f"Extrahiere die Filmdaten:\n\n{film_desc_input.value}",
                },
            ],
            temperature=temperature_slider.value,
            max_retries=3,
        )
    return (instructor_result,)


@app.cell
def _(instructor_result, mo):
    mo.json(instructor_result.model_dump())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Zusammenfassung

    | Technik | Wann einsetzen |
    |---------|----------------|
    | **Klar & Direkt** | Immer – Grundlage jedes guten Prompts |
    | **System-Prompt** | Wenn das Modell eine Rolle, Regeln oder einen Stil annehmen soll |
    | **XML-Tags** | Wenn Daten und Anweisungen klar getrennt werden müssen |
    | **Few-Shot** | Wenn Format oder Stil durch Beispiele einfacher zu zeigen als zu beschreiben ist |
    | **Chain of Thought** | Bei komplexen Schlussfolgerungen oder mehrstufigen Aufgaben |
    | **JSON / Instructor** | Wenn die Ausgabe programmatisch weiterverarbeitet wird |

    ## Allgemeine Tipps

    - Systematisches Iterieren und Testen der Prompts
    - Variation der Temperature: 0 für konsistente Ausgaben, höher für kreative Aufgaben
    - Konsequente Verwendung von XML-Tags zur Trennung von Anweisungen und Daten
    - Instructor ist JSON Mode fast immer vorzuziehen, wenn Typsicherheit benötigt wird
    """)
    return


if __name__ == "__main__":
    app.run()
