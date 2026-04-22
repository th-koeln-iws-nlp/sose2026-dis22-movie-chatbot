0. Legt in einer extra Datei einen System Prompt für euren Chatbot an und ladet ihn in das jeweilige LLM. Ihr könnt euch auch verschiedene Rollen ausdenken und die Nutzer*innen diese auswählen lassen.
1. Erstellt eine Vektordatenbank. Das kann qdrant mit einem Free Trial Client oder Chroma DB mit persistent Storage oder eine andere Vektordatenbank sein.
2. Entscheidet euch für ein Embedding Model. Zum Beispiel Gemini, OpenAI oder ein lokales über HuggingFace bzw `sentence-transformers`. **Wichtig:** Das Embedding Model ist dann später auch das welches für RAG benutzt wird.
3. Füllt die Vektordatenbank mit Filmen. Der Plot des Films ist dabei der Vektor. Die Metadaten sind mindestens: Filmtitel, Erscheinungsjahr und Genre. Mögliche Filmdatenbanken seht ihr unten. Als Grundlage könnt ihr das [embeddings](./notebooks/embeddings.py) marimo notebook nehmen.
4. Ändert eure "Movies" Seite in Streamlit so ab, dass die Filme aus der Vektordatenbank gezeigt werden. Achtet auf so etwas wie Pagination, damit nicht alle Filme gleichzeitig geladen und angezeigt werden.
5. Fügt der "Movies" Seite eine Suchleiste hinzu. Die Suchergebnisse basieren auf einer semantischen Suchen in der Vektordatenbank. Die angezeigten Filme auf der "Movies" Seite sind dann die Sucheregebnisse. Zeigt auch die Similarity Score zur Suchanfrage mit an.

**Bonus**: Fügt einen neuen Menüpunkt "Recommendations" hinzu. Auf dieser Seite, kann man einen Film aus der Vetkordatenbank auswählen und bekommt darauf basierend ähnliche Filme vorgeschlagen.


**Mögliche Filmdatenbanken**

- Die im Tutorial verwendeten TMDB Top 5000 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- IMDb Top 1000 auf Kaagle https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows
- Knapp 10000 IMDB Filme https://www.kaggle.com/datasets/ashpalsingh1525/imdb-movies-dataset
- Wikipedia Daten, aufgesplittet nach Jahrzehnten https://github.com/prust/wikipedia-movie-data
- Kombiniert aus https://www.kaggle.com/datasets/yashgupta24/48000-movies-dataset und https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots
- Weitere Datensätze https://www.kaggle.com/datasets?tags=2303-Movies+and+TV+Shows