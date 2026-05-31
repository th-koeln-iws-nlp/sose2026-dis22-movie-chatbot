# Meilenstein 04: Tool Use & Agenten

Baut euren Film-Chatbot zu einem Agenten aus. Das LLM entscheidet selbst, welche Tools es aufruft.

1. Implementiert ein RAG-Tool für die semantische Suche in eurer Vektordatenbank. Das Tool soll neben dem Query alle Metadaten als optionale Filterparameter unterstützen, die ihr in der Datenbank habt, z.B. Genre, Erscheinungsjahr, Bewertung, Sprache, Laufzeit.

2. Implementiert ein Tool, das aktuelle Film-Charts von der TMDB API abruft. Hier könnt auch ihr auch noch andere Funktionalitäten der TMDB API (zB konkreter Film) nutzen. 

3. Implementiert ein Tool, das das aktuelle Kölner Kinoprogramm abruft. Die Datenquelle ist [koeln.de/kino](https://www.koeln.de/kino/). Das Tool soll zumindest nach Datum filterbar sein. Alles weitere, z.B. nach bestimmten Kinos filtern, ist euch überlassen.

   **Hinweis zur Umsetzung:** Die Seite ist server-side gerendert und enthält strukturierte Schema.org-Daten. Ein einfaches HTTP-Request + HTML-Parsing reicht aus, kein Browser-Automation nötig. Die Seite zeigt immer nur ein paar Tage an, das sollte das Tool transparent machen.

4. Integriert den Agenten in euren Chatbot. Zeigt in der Antwort an, welche Tools aufgerufen wurden.

**Denkt außerdem** schon mal daran, dass ihr bis zur Abschlusspräsentation ein weiteres Tool entwickeln sollt, das im Seminar nicht besprochen wurde. Ihr seid dabei völlig frei.
