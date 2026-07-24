# 📚 Local Academic Notes AI Assistant (RAG)

Lokalny asystent AI służący do inteligentnej analizy i odpytywania dokumentów PDF. Projekt wykorzystuje architekturę **RAG (Retrieval-Augmented Generation)**, działając w **100% lokalnie i prywatnie** bez wysyłania danych do zewnętrznych chmur czy płatnych API.

> **Projekt zrealizowany w ramach portfolio (Akademia WIT).**

---

## 🛠️ Stack Technologiczny

* **Language:** Python 3.12+
* **UI Framework:** [Streamlit](https://streamlit.io/)
* **AI Orchestration:** [LangChain](https://www.langchain.com/) (LangChain Expression Language - LCEL)
* **LLM & Embeddings (Local):** [Ollama](https://ollama.com/) (`llama3.2` + `nomic-embed-text`)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/)
* **PDF Processing:** `pdfplumber`

---

## 🏗️ Jak działa aplikacja? (Architektura RAG)

1. **Przetwarzanie PDF:** Dokument PDF jest wczytywany i dzielony na mniejsze fragmenty tekstu (*chunks*) za pomocą `RecursiveCharacterTextSplitter`.
2. **Wektorowanie (Embeddings):** Każdy fragment tekstu jest konwertowany na wielowymiarowy wektor za pomocą modelu `nomic-embed-text`.
3. **Pamięć (Vector Database):** Wektory wraz z tekstami źródłowymi są zapisywane w lokalnej bazie `ChromaDB`.
4. **Wyszukiwanie kontekstu (Retrieval):** Gdy użytkownik zadaje pytanie, baza ChromaDB odnajduje 3 najbardziej powiązane tematycznie fragmenty z dokumentu.
5. **Generowanie odpowiedzi:** Model `llama3.2` otrzymuje pytanie użytkownika wraz z odnalezionym kontekstem, tworząc precyzyjną i popartą notatkami odpowiedź.

---

## 🚀 Instrukcja Uruchomienia Lokalnie

### 1. Wymagania Wstępne
Upewnij się, że masz zainstalowany program **[Ollama](https://ollama.com/)**.
Pobierz wymagane modele lokalne w terminalu:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Klonowanie Repozytorium i Środowisko

# 1. Sklonuj repozytorium
```bash
git clone [https://github.com/Alienman89/SmartPDF.git](https://github.com/Alienman89/SmartPDF.git)
cd SmartPDF
```
# 2. Utwórz i aktywuj środowisko wirtualne
```bash
python -m venv .venv
```
# Windows (PowerShell):
```bash
.\.venv\Scripts\Activate.ps1
```
# Linux / macOS:
```bash
source .venv/bin/activate
```
# 3. Zainstaluj wymagane pakiety
```bash
pip install -r requirements.txt
```

### 3. Uruchomienie Aplikacji

```bash
python -m streamlit run app.py
```
