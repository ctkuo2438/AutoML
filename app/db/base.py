'''
Importing the declarative_base function from SQLAlchemy.
This function is used to create a base class for all ORM models.
ORM (Object-Relational Mapping) allows us to define database tables as Python classes.
'''
from sqlalchemy.ext.declarative import declarative_base

'''
Create a base class for ORM models.
All database models will inherit from this Base class.
It provides the foundation for defining tables and mapping them to Python classes.
'''
Base = declarative_base()