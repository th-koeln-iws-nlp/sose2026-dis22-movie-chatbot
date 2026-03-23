# Film & Kino Agent

Dies ist das GitHub Repository für das DIS 22 Projekt im Sommersemester 2026.

[Kick-Off Slides](slides/kick_off.pdf)

## Form der Prüfung

Die Anforderungen zum Bestehen des Kurses sind:

- Gut dokumentierter und funktionsfähiger Code
- Insbesondere funktionierende Implementationen der Meilensteine
- Innerhalb des Codes: Kurze Erläuterungen zu den Ergebnissen
- Mindestens ein neues LLM Tool, das im Seminar nicht besprochen wurde

### Abschlusspräsentation

Jedes Teammitglied stellt ca. 10 Minuten einen Teil des Projekts vor (ohne Folien). Die Aufteilung erfolgt selbstständig innerhalb des Teams. Nach jeder Vorstellung gibt es eine kleine Fragerunde.

Termin: **01.07.2026**

## Termine

| KW | Datum        | Thema                                          |
|----|--------------|------------------------------------------------|
| 16 | 15.04.2026   | Kick-Off, Projektvorstellung, Setup            |
| 17 | 22.04.2026   | UI bauen, LLM-API-Anbindung, Prompt Engineering |
| 18 | 29.04.2026   | Vektordatenbanken & Embeddings                 |
| 19 | 06.05.2026   | Filmdatenbank vektorisieren                    |
| 20 | 13.05.2026   | *Home Office*                                  |
| 21 | 20.05.2026   | RAG: Theorie + Implementation (Projektwoche)   |
| 22 | 27.05.2026   | *Home Office*                                  |
| 23 | 03.06.2026   | Tool Use & Agenten                             |
| 24 | 10.06.2026   | Agenten-Implementation + Theorie: Evaluation   |
| 25 | 17.06.2026   | Gegenseitiger Nutzertest + Feinschliff         |
| 26 | 24.06.2026   | Feinschliff, Bugfixes                          |
| 27 | 01.07.2026   | Abschlusspräsentationen                        |

## Die Teams

*(wird ergänzt)*

## Architektur

Die Architektur des Chatbots ist in der folgenden Grafik dargestellt:

![Chatbot Architecture](img/architecture.png)

## Setup

- `uv` installieren

```bash
# Abhängigkeiten installieren
uv sync

# API Keys in .env eintragen
# GEMINI_API_KEY=...
# OPEN_WEB_UI_API_KEY=...

# Chatbot starten
just chatbot
```

## Linksammlung

### Allgemein zu NLP

- <https://e2eml.school/transformers.html>
- <https://web.stanford.edu/~jurafsky/slp3/>
- <https://lena-voita.github.io/nlp_course.html>
- <https://github.com/keon/awesome-nlp>

### Large Language Models

- [Anti-hype LLM reading list](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)
- [Gemini Cookbook](https://github.com/google-gemini/cookbook)
- <https://github.com/mlabonne/llm-course>
- [VIDEO What is GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M)
- [VIDEO A Hackers' Guide to Language Models](https://www.youtube.com/watch?v=jkrNMKz9pWU)

### Retrieval Augmented Generation

- <https://github.com/langchain-ai/rag-from-scratch>
- <https://www.promptingguide.ai/research/rag>
- <https://www.pinecone.io/learn/>
- <https://realpython.com/build-llm-rag-chatbot-with-langchain/>
