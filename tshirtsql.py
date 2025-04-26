
db_user="username"
db_password="passowrd"

#db_password="passowrd"
#db_host="aws"
db_host="localhost"
db_port=1234
db_name="atliq_tshirts"
from langchain_community.utilities.sql_database import SQLDatabase
db=SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",)

print(db.dialect)
print(db.get_usable_table_names())
print(db.table_info)

from langchain.chains import create_sql_query_chain
#from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key="your_api_key")

generate_query= create_sql_query_chain(llm, db)
query = generate_query.invoke({"question": "list of all brands?"} )


#query="SELECT * FROM atliq_tshirts.t_shirts;"
print(query)


import re

def extract_sql(query):
    """
    Extracts the SQL query from LLM response.
    Looks for the line starting with SQLQuery: or just finds the SELECT.
    """
    match = re.search(r'SQLQuery:\s*(SELECT[\s\S]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # fallback: find a raw SELECT if "SQLQuery:" isn't there
    match = re.search(r'(SELECT[\s\S]+)', query, re.IGNORECASE)
    return match.group(1).strip() if match else None

#llm_output = """
#Question: what is the price one nike tshirt?
#SQLQuery: SELECT `price` FROM `products` WHERE `brand` = 'Nike' AND `category` = 'T-Shirt' LIMIT 1;
#"""

query_new = extract_sql(query)
print(query_new)

#cursor.execute(query_new)
#results = cursor.fetchall()
#print(results)


from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


execute_query = QuerySQLDatabaseTool(db=db)
results=execute_query.invoke(query_new)

print("Query Results:\n", results)

chain= generate_query | execute_query
chain.invoke({"question" : "list of all brands?"} )

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


answer_prompt = PromptTemplate.from_template(
    """Given the followinf user question, corresponding sql query, and sql result, answer the user questiion.
    
Question : {question}
SQl Query : {query}
SQL Result: {result}
Answer: """

)

rephrase_answer = answer_prompt | llm | StrOutputParser()

chain= (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query")| execute_query

    )
    | rephrase_answer
)

examples = [


    {'input': "How many t-shirts do we have left for Nike in XS size and white color?",
     'query': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
     #'SQLResult': "Result of the SQL query",
     #'Answer': "91"
     },
    {'input': "How much is the total price of the inventory for all S-size t-shirts?",
     'query': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
     #'SQLResult': "Result of the SQL query",
     #'Answer': "22292"
     },
    {
        'input': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?",
        'query': """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
      #  'SQLResult': "Result of the SQL query",
       # 'Answer': "16725.4"
         },
    {
        'input': "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?",
        'query': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
        #'SQLResult': "Result of the SQL query",
        #'Answer': "17462"
        },
    {'input': "How many white color Levi's shirt I have?",
     'query': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
     #'SQLResult': "Result of the SQL query",
     #'Answer': "290"
     },
    {
        'input': "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
        'query': """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Nike' and size="L"
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
      #  'SQLResult': "Result of the SQL query",
      #  'Answer': "290"
        }
]

chain.invoke({"question" : "list of all brands?"} )

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, \
    PromptTemplate

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    # input_variables=["input","top_k"],
    input_variables=["input"],
)
print(few_shot_prompt.format(input="How many nike tshirts are there?"))

#openai_api_key="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"
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma()
vectorstore.delete_collection()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    vectorstore_cls=Chroma,
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    k=2,
    input_keys=["input"],)
example_selector.select_examples({"input": "how many nike tshirts we have?"})
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input","top_k"],
)
print(few_shot_prompt.format(input="How many nike tshirts are there?"))


from langchain_core.prompts import ChatPromptTemplate

final_prompt = ChatPromptTemplate(
    input_variables=["input","top_k","table_info"],
    messages=[
        ("system", "You are a MySQL expert. Given an input question, return ONLY the SQL query to answer it.Do NOT include any explanations, just output the raw SQL starting with SELECT or appropriate "
                   "keyword.  Unless otherwise specified.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
        few_shot_prompt,
        ("human", "{input}")
    ]
)
(final_prompt.format(input="How many t-shirts do we have left for Nike in XS size and white color??",table_info=db.table_info))

generate_query = create_sql_query_chain(llm, db, final_prompt)
chain = (
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
)
your_question="If we sell all Adidas medium-size blue t-shirts today with discounts applied, how much revenue will we generate?"
print(your_question)
response= chain.invoke({"question": your_question})
print(response)

