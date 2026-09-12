from sqlalchemy import create_engine

# SQL Server configuration
SERVER = r"localhost\SQLEXPRESS"
DATABASE = "FraudDetectionDB"

connection_string = (
    "mssql+pyodbc://@"
    + SERVER
    + "/"
    + DATABASE
    + "?driver=ODBC+Driver+17+for+SQL+Server"
    + "&trusted_connection=yes"
)

engine = create_engine(
    connection_string,
    pool_pre_ping=True
)

print("SQL Server engine created successfully")