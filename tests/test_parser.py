import pytest
from pathlib import Path
from src.rag.parser import PolicyParser
from src.config import settings


def test_parser_chunks_integrity():
    pdf_path = settings.policy_pdf_path
    assert pdf_path.exists(), f"Policy PDF missing at {pdf_path}"

    parser = PolicyParser(pdf_path)
    chunks = parser.parse()

    assert len(chunks) > 20, "Parser should extract multiple structural chunks"

    for chunk in chunks:
        assert chunk.chunk_id.startswith("POL-")
        assert chunk.page >= 1
        assert len(chunk.section) > 0
        assert len(chunk.text) > 20
        assert isinstance(chunk.keywords, list)
