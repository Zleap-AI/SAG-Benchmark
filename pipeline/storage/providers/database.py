"""Structured database backend implementations."""

from sqlalchemy import text

from pipeline.db.base import close_database, get_engine


class _SqlAlchemyDatabaseStore:
    backend_name = "sqlalchemy"

    async def health_check(self) -> bool:
        try:
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await close_database()


class MySqlDatabaseStore(_SqlAlchemyDatabaseStore):
    """MySQL structured data backend."""

    backend_name = "mysql"


class OceanBaseDatabaseStore(_SqlAlchemyDatabaseStore):
    """OceanBase structured data backend in MySQL-compatible mode."""

    backend_name = "oceanbase"
