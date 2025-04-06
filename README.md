# 🧠 Natural Language to SQL Query System for T-Shirt Inventory

This project allows you to ask **natural language questions** about a MySQL-based t-shirt inventory database — and get accurate answers powered by **LangChain**, **Gemini 1.5**, and **MySQL**.

## 🚀 Features

- ✅ Ask questions like “How many Nike t-shirts are there?”
- ✅ Automatically generates SQL queries from your questions
- ✅ Connects to a MySQL database (local or cloud)
- ✅ Applies discounts and calculates revenue dynamically
- ✅ Uses few-shot and semantic similarity examples to improve query accuracy

## 🛠 Tech Stack

- Python 🐍
- LangChain
- Google Gemini 1.5 Flash
- MySQL (localhost or AWS RDS)
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Chroma Vector Store

## 💽 Database Schema

### `t_shirts` Table

| Column         | Type                     |
|----------------|--------------------------|
| t_shirt_id     | INT (PK, AUTO_INCREMENT) |
| brand          | ENUM('Van Huesen', 'Levi', 'Nike', 'Adidas') |
| color          | ENUM('Red', 'Blue', 'Black', 'White') |
| size           | ENUM('XS', 'S', 'M', 'L', 'XL') |
| price          | INT (10 to 50)           |
| stock_quantity | INT                      |

### `discounts` Table

| Column        | Type                    |
|---------------|-------------------------|
| discount_id   | INT (PK, AUTO_INCREMENT)|
| t_shirt_id    | INT (FK → t_shirts)     |
| pct_discount  | DECIMAL(5,2) (0-100)    |

## 🧪 Example Questions You Can Ask

- "How many Nike t-shirts are there?"
- "What is the total price of all S-size t-shirts?"
- "If we sell all Adidas medium-size blue t-shirts today with discounts applied, how much revenue will we generate?"
- "Which color and size combination has the highest stock?"

## 🧰 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/azibkhan24/tshirtsql.git
cd tshirtsql
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

```

### 3. Setup MySQL Database

- Run the provided `db_creation_atliq_t_shirts.sql` to create tables and sample data
- Update your `db_user`, `db_password`, and `db_host` in the script if needed

### 4. Add Your Gemini API Key

Set your Google API Key:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key="YOUR_GOOGLE_API_KEY"
)
```

## 📈 Output Example

```
Question: If we sell all Adidas medium-size blue t-shirts today with discounts applied, how much revenue will we generate?

Generated SQL:
SELECT SUM(price * stock_quantity * ((100 - COALESCE(d.pct_discount, 0)) / 100)) AS revenue FROM ...

Answer: The total revenue after applying discounts would be ₹1,356.25.
```

## 🧠 Credits

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Google Gemini](https://ai.google.dev/)
- Embeddings by HuggingFace

---

⭐️ Feel free to fork, contribute, and ⭐️ the repo!
