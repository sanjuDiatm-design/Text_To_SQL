# 🤖 Text-to-SQL Query Generator & Runner

An intelligent, interactive Streamlit application powered by LangChain and Groq LLMs that translates natural language questions into accurate SQL queries, executes them on SQLite databases, and visualizes the results.

---

## 🚀 Features

- **Natural Language to SQL:** Ask questions in plain English and automatically get optimized SQL queries.
- **Automated Execution:** Runs the generated queries against SQLite databases and displays query outputs in clean tables.
- **Custom Database Support:** Query the built-in student database or upload custom SQLite databases.
- **Fast Inference:** Powered by Groq for near-instant response times.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM & Orchestration:** [LangChain](https://www.langchain.com/), [Groq API](https://groq.com/)
- **Database:** SQLite3
- **Data Manipulation:** Pandas

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Initialize the database
```bash
python database.py
```

### 6. Run the application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
