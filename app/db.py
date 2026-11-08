"""Task 5 - the prediction trail: one permanent row per verdict (what, when, which model, how confident)."""
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, create_engine,
                        desc, func, select, text)

metadata = MetaData()
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),   # WHEN (UTC)
    Column("filename", String(255)),                                 # sanitized name only - no image stored
    Column("file_size", Integer),
    Column("width", Integer),
    Column("height", Integer),
    Column("image_format", String(16)),
    Column("label", String(32), nullable=False),                      # WHAT was decided
    Column("confidence", Float, nullable=False),                      # HOW confidently
    Column("p_ai", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_version", String(128), nullable=False),             # BY WHICH model
    Column("inference_ms", Float),
    Column("source", String(16), nullable=False),                     # web | api
)


class Database:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)
        metadata.create_all(self.engine)            # creates the table on first start

    def save(self, **row):
        row.setdefault("created_at", datetime.now(timezone.utc))
        with self.engine.begin() as conn:            # begin() = commit automatically
            return conn.execute(predictions.insert().values(**row)).inserted_primary_key[0]

    def recent(self, limit=100):
        with self.engine.connect() as conn:
            q = select(predictions).order_by(desc(predictions.c.id)).limit(limit)
            return conn.execute(q).mappings().all()

    def count(self):
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(predictions)).scalar_one()

    def ping(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
