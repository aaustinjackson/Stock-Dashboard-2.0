from sqlalchemy import create_engine

# Replace these with your PostgreSQL credentials
username = "your_username"
password = "your_password"
host = "localhost"
port = "5432"
database = "stocks_db"

# Create engine
engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

# Test connection by reading table names
print(engine.table_names())