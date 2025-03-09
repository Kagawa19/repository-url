
from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Expert(Base):
    __tablename__ = 'experts_expert'

    id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    designation = Column(String)
    theme = Column(String)
    unit = Column(String)
    contact_details = Column(String)
    knowledge_expertise = Column(JSON)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class Resource(Base):
    __tablename__ = 'resources_resource'

    id = Column(Integer, primary_key=True)
    doi = Column(String)
    title = Column(String, nullable=False)
    abstract = Column(String)
    summary = Column(String)
    domains = Column(JSON)
    topics = Column(JSON)
    description = Column(String)
    expert_id = Column(Integer)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)