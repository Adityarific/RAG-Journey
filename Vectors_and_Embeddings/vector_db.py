from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Sample dataset formatted for LangChain RAG pipelines
docs = [
    Document(
        page_content="TechVibe offers a discretionary Unlimited Paid Time Off (PTO) policy for all full-time employees. Employees must submit PTO requests through the HR portal at least two weeks in advance for planned absences exceeding three consecutive business days. There is no cash payout for unused PTO upon separation of employment because days do not accrue. Managers approve requests based on team coverage and current project deadlines. Peak business season runs from October 1 to December 15, during which PTO approvals are strictly limited.",
        metadata={
            "source": "HR-001",
            "title": "Unlimited PTO and Vacation Policy",
            "category": "HR"
        }
    ),
    Document(
        page_content="Full-time remote employees qualify for a one-time home office stipend of up to $500 USD. Eligible items include ergonomic chairs, desks, external monitors, and keyboards. Computer hardware like laptops and mice are provided directly by IT and cannot be purchased for reimbursement. Employees must submit receipts via the Expenseify platform within 30 days of purchase. The stipend resets if an employee undergoes a lateral department transfer that requires a different physical setup.",
        metadata={
            "source": "IT-204",
            "title": "Home Office Equipment Reimbursement",
            "category": "IT Support"
        }
    ),
    Document(
        page_content="OmniQuery is TechVibe's flagship data analytics tool. It uses natural language processing to generate SQL queries from plain English prompts. Version 4.2 introduces the 'DataGuard' feature, which automatically masks Personally Identifiable Information (PII) before running queries against cloud databases. OmniQuery natively integrates with Snowflake, AWS Redshift, and Google BigQuery. It requires a minimum cluster size of 2 nodes to maintain real-time dashboard updates.",
        metadata={
            "source": "PROD-772",
            "title": "OmniQuery AI Analytics Dashboard",
            "category": "Product"
        }
    ),
    Document(
        page_content="Users encounter Error Code 403 when their security token expires or lacks necessary IAM permissions. To resolve this, first ask the user to log out, clear their browser cache, and log back in. If the error persists, the system administrator must verify that the user's role has the 'Reader' permission enabled in the TechVibe Cloud Console. For Snowflake connections, ensure the specific warehouse network policy allows traffic from TechVibe's static IP range (192.0.2.50/32).",
        metadata={
            "source": "SUPP-105",
            "title": "Resolving Error Code 403 on OmniQuery",
            "category": "Customer Support"
        }
    )
]

# This runs entirely on your local machine for free
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create Chroma vector store and inject documents directly
vectors = Chroma.from_documents(
    documents= docs,
    embedding = embeddings ,
    persist_directory="./my_rag_db2"
)

print('data injected into database')
