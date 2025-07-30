"""Database initialization and management."""

from typing import AsyncGenerator, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from ..core import get_logger, settings


logger = get_logger(__name__)

Base = declarative_base()

# Global database engine and session factory
engine = None
SessionLocal = None


async def init_database():
    """Initialize database connection and create tables."""
    global engine, SessionLocal
    
    try:
        # Create async engine
        database_url = settings.database_url
        
        # Handle PostgreSQL vs SQLite URLs
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        elif database_url.startswith("sqlite://"):
            database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        
        # Engine configuration based on database type
        engine_kwargs: Dict[str, Any] = {"echo": settings.debug}
        
        # Only set pool parameters for non-SQLite databases
        if not database_url.startswith("sqlite"):
            engine_kwargs.update({
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
            })
        
        engine = create_async_engine(database_url, **engine_kwargs)
        
        # Create session factory
        SessionLocal = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if SessionLocal is None:
        await init_database()
    
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_database():
    """Close database connections."""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")
