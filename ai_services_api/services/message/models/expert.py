from sqlalchemy import Column, Integer, String, Boolean, ARRAY, DateTime, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Expert(Base):
    __tablename__ = "experts_expert"  # Matches your existing table name

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    designation = Column(String(255))
    theme = Column(String(255))
    unit = Column(String(255))
    contact_details = Column(String(255))
    knowledge_expertise = Column(JSON)
    orcid = Column(String(255))
    domains = Column(ARRAY(String))
    fields = Column(ARRAY(String))
    subfields = Column(ARRAY(String))
    password = Column(String(255))
    is_superuser = Column(Boolean, default=False)
    is_staff = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime(timezone=True))
    date_joined = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    bio = Column(Text)
    email = Column(String(200))
    middle_name = Column(String(200))
