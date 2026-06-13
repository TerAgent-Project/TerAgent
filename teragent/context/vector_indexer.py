# teragent/context/vector_indexer.py
import logging
import os
from typing import Any

import lancedb

from teragent.utils.token_counter import estimate_tokens

logger = logging.getLogger(__name__)

TABLE_NAME = "code_symbols"
MAX_CODE_TOKENS = 8000
MAX_CODE_CHARS = 32_000


class VectorIndexer:
    """Vector-based semantic search over code symbols using LanceDB.

    Falls back gracefully when the embedding API is unavailable: all
    write operations silently skip and searches return empty results.
    """

    def __init__(
        self,
        workspace_root: str,
        embedding_api_url: str | None = None,
        embedding_api_key: str | None = None,
    ) -> None:
        db_path = os.path.join(workspace_root, ".agent", "vectors")
        self.db = lancedb.connect(db_path)
        self.table: lancedb.table.Table | None = None
        self.api_url = embedding_api_url or os.getenv("EMBEDDING_API_URL")
        self.api_key = embedding_api_key or os.getenv("EMBEDDING_API_KEY")
        self._table_initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_table(self) -> None:
        """Lazily ensure the LanceDB table exists before writes/queries."""
        if self._table_initialized:
            return
        try:
            existing_tables = self.db.table_names()
            if TABLE_NAME in existing_tables:
                self.table = self.db.open_table(TABLE_NAME)
            # If table doesn't exist yet, it will be created on first add_code
            self._table_initialized = True
        except Exception as e:
            logger.warning(f"Failed to check existing tables: {e}")
            # Don't set _table_initialized — allow retry on next call

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """调用外部 API 获取向量表示"""
        if not self.api_url:
            raise RuntimeError("Embedding API URL not configured")

        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for embedding API calls")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": "text-embedding-3-small",  # 默认使用 OpenAI 规格
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data.get("data", [])]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def add_code(self, symbol_name: str, code: str, file_path: str) -> None:
        """向量化并存储代码片段"""
        if not code.strip():
            return

        # 截断超长代码
        if estimate_tokens(code) > MAX_CODE_TOKENS:
            code = code[:MAX_CODE_CHARS]

        try:
            await self._ensure_table()
            embeddings = await self._get_embeddings([code])
            if not embeddings:
                logger.warning(f"Empty embeddings returned for {symbol_name}, skipping")
                return
            data = [
                {
                    "vector": embeddings[0],
                    "symbol": symbol_name,
                    "file": file_path,
                    "code": code,
                }
            ]

            if self.table is None:
                self.table = self.db.create_table(TABLE_NAME, data)
            else:
                self.table.add(data)
        except RuntimeError as e:
            logger.warning(f"Embedding API unavailable, skipping add_code: {e}")
        except Exception as e:
            logger.error(f"Failed to add vector for {symbol_name}: {e}")

    async def delete_by_symbol(self, symbol_name: str) -> int:
        """Delete all vector entries for the given symbol name.

        Args:
            symbol_name: The symbol to remove from the index.

        Returns:
            1 on success (LanceDB delete does not return count),
            0 on failure.
        """
        await self._ensure_table()
        if self.table is None:
            return 0
        try:
            # Escape special characters to prevent injection
            safe_name = symbol_name.replace("\\", "\\\\").replace('"', '\\"')
            self.table.delete(f'symbol = "{safe_name}"')
            logger.info(f"Deleted vector entries for symbol: {symbol_name}")
            return 1  # LanceDB delete doesn't return count
        except Exception as e:
            logger.error(f"Failed to delete vectors for {symbol_name}: {e}")
            return 0

    async def index_file_symbols(self, symbols: list[dict]) -> int:
        """Batch-index all symbols from CodeIndexer output.

        Args:
            symbols: A list of dicts, each with keys ``name``,
                ``signature``, ``file_path``, and optionally ``source_code``.

        Returns:
            The number of symbols successfully indexed.
        """
        indexed = 0
        for sym in symbols:
            # Use full source code when available, fall back to signature
            code = sym.get("source_code", "") or sym.get("signature", "")
            if not code.strip():
                continue
            await self.add_code(sym["name"], code, sym.get("file_path", ""))
            indexed += 1
        logger.info(f"Batch-indexed {indexed}/{len(symbols)} symbols into vector store")
        return indexed

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def semantic_search(self, query: str, top_k: int = 5) -> list[str]:
        """语义检索相关代码片段"""
        await self._ensure_table()
        if self.table is None:
            return []

        try:
            query_embeddings = await self._get_embeddings([query])
            results = self.table.search(query_embeddings[0]).limit(top_k).to_list()
            return [
                f"// File: {r['file']}, Symbol: {r['symbol']}\n{r['code']}"
                for r in results
            ]
        except RuntimeError as e:
            logger.warning(f"Embedding API unavailable for search: {e}")
            return []
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Return statistics about the vector index.

        Returns:
            A dict with keys ``table_exists``, ``row_count``,
            ``api_configured``.
        """
        await self._ensure_table()
        row_count = 0
        if self.table is not None:
            try:
                row_count = self.table.count_rows()
            except Exception as e:
                logger.debug(f"Failed to count rows: {e}")
                row_count = -1  # Unknown
        return {
            "table_exists": self.table is not None,
            "row_count": row_count,
            "api_configured": self.api_url is not None,
        }
