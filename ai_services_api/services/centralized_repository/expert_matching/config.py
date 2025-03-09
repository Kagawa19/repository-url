import os

class Config:
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/mydatabase')
    EXPERTISE_CSV = os.getenv('EXPERTISE_CSV', 'path/to/expertise.csv')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')