# Meilenstein 05: Gegenseitiger Chatbot-Test

Ihr baut ein Testset und testet damit die Agenten der anderen Teams.

1. Erstellt ein Testset als Tabelle mit mindestens 15 Testfällen. Die Testfälle sollen RAG und Tool Use abdecken. Mindestens diese Spalten:

   | Anfrage | Typ | Erwartete Ausgabe | Generierte Ausgabe | Kommentar / Pass-Fail |
   |---------|-----|-------------------|--------------------|-----------------------|
   | "Empfiehl Mafia-Dramen aus den 2000ern" | RAG + Filter | The Departed, ... | (leer) | (leer) |
   | "Was läuft gerade?" | Tool (Charts) | aktuelle TMDB-Liste | (leer) | (leer) |
   | "Trailer zu Avatar" | nicht unterstützt | ehrliche Absage | (leer) | (leer) |

2. Achtet auf diese Hinweise beim Bauen des Testsets:
   - **Echte Anfragen:** Keine generischen Lehrbuch-Queries. Überlegt, was Nutzer wirklich fragen würden, und streut Edge Cases ein.
   - **Erwartete Ausgabe nicht zu eng fassen:** Bei RAG gibt es selten die eine Antwort. Beschreibt, was eine gute Antwort enthalten muss, z.B. "mindestens 2 Crime-Filme aus 2000-2009".
   - **Binär bewerten:** Pass/Fail mit Begründung, keine 1-5-Skala. "Fail, weil Film X halluziniert wurde" ist wertvoller als ein bloßes Fail.
   - **Tool-Pfad mitdenken:** Notiert auch, welches Tool aufgerufen werden sollte, nicht nur den finalen Text.
   - **Diverse Dimensionen:** verschiedene Features (RAG, Charts, Filter), Szenarien (Treffer, kein Treffer, mehrdeutig) und Personas (Genre-Fan, Gelegenheitsgucker).

3. Testet das Testset zuerst auf eurem eigenen Chatbot und füllt die Tabelle aus.

4. Am 17.6  testet ihr damit die Chatbots der anderen Teams und dokumentiert die Ergebnisse.
