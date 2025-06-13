from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from src.database.psql_schema import User, Report, LabResult, MedicalImage, Embeddings
from src.utils.settings import SETTINGS

Base = declarative_base()


class PostgreSQL:
    """PostgreSQL connection manager using SQLAlchemy"""

    def __init__(self):
        """Initialize PostgreSQL connection manager"""
        self.engine = None
        self.async_session = None

    async def init_psql_db(self) -> None:
        """Initialize PostgreSQL connection"""
        if self.engine is None:
            self.engine = create_async_engine(
                SETTINGS.DATABASE_URL,
                echo=False,  # reduce logging
            )

            # Create tables if they do not exist
            # Do nothing if the tables already exist with the same schema
            async with self.engine.begin() as conn:
                tables = [
                    User.__table__,
                    Report.__table__,
                    LabResult.__table__,
                    MedicalImage.__table__,
                    Embeddings.__table__,
                ]
                await conn.run_sync(Base.metadata.create_all, tables)

        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def close_psql_db(self) -> None:
        """Close database connection"""
        if self.engine is not None:
            await self.engine.dispose()

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get SQLAlchemy async session"""
        if self.async_session is None:
            await self.init_psql_db()
        assert self.async_session is not None
        async with self.async_session() as session:
            try:
                yield session
            finally:
                await session.close()


PSQL = PostgreSQL()
