import psycopg

from organization_rag.config import TOP_K
from organization_rag.embeddings import get_embedding


DB_URL = "postgresql://raguser:ragpassword@localhost:5433/organization_rag"
TABLE_NAME = "document_chunks"


class PostgresVectorStore:
    def __init__(self):
        self.conn = psycopg.connect(DB_URL)
        self._create_table()

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(768) NOT NULL,
                    metadata JSONB NOT NULL
                )
                """
            )

        self.conn.commit()

    def reset(self):
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

        self.conn.commit()
        self._create_table()

    def add_chunks(self, chunks):
        with self.conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk.text)

                metadata = {
                    key: str(value)
                    for key, value in chunk.metadata.items()
                    if value is not None
                }

                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME}
                        (id, content, embedding, metadata)
                    VALUES
                        (%s, %s, %s::vector, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        f"chunk-{i}",
                        chunk.text,
                        str(embedding),
                        psycopg.types.json.Jsonb(metadata),
                    ),
                )

        self.conn.commit()
        return len(chunks)

    def search(self, query, department=None, top_k=TOP_K):
        query_embedding = get_embedding(query)

        sql = f"""
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM {TABLE_NAME}
        """

        params = [str(query_embedding)]

        if department:
            sql += " WHERE metadata->>'department' = %s"
            params.append(department)

        sql += """
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        params.extend([str(query_embedding), top_k])

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return rows

    def close(self):
        self.conn.close()
