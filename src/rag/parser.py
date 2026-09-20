import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader


class PolicyChunk:
    def __init__(
        self,
        chunk_id: str,
        source: str,
        page: int,
        section: str,
        clause: str,
        text: str,
        keywords: Optional[List[str]] = None
    ):
        self.chunk_id = chunk_id
        self.source = source
        self.page = page
        self.section = section
        self.clause = clause
        self.text = text
        self.keywords = keywords or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "clause": self.clause,
            "text": self.text,
            "keywords": self.keywords,
        }


class PolicyParser:
    """
    Hierarchical parser for insurance policy documents.
    Preserves structural metadata (page numbers, section headings, clause identifiers)
    and performs semantic chunking by policy articles rather than fixed token counts.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Policy PDF not found at {self.pdf_path}")

    def parse(self) -> List[PolicyChunk]:
        reader = PdfReader(str(self.pdf_path))
        raw_pages: List[Dict[str, Any]] = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Normalize whitespace while preserving line boundaries
            cleaned_text = re.sub(r"[ \t]+", " ", text).strip()
            raw_pages.append({
                "page_number": page_idx + 1,
                "text": cleaned_text
            })

        chunks = self._chunk_hierarchically(raw_pages)
        return chunks

    def _chunk_hierarchically(self, pages: List[Dict[str, Any]]) -> List[PolicyChunk]:
        chunks: List[PolicyChunk] = []
        current_section = "Preamble & General Terms"
        chunk_counter = 1

        # Known section markers in health insurance policy wordings
        section_patterns = [
            (r"(?i)(?:section|part)\s+[1I]\s*[:\-–]?\s*(?:scope of cover|benefits|coverage)", "Scope of Cover"),
            (r"(?i)(?:section|part)\s+[2II]\s*[:\-–]?\s*(?:waiting periods?|time exclusions?)", "Waiting Periods"),
            (r"(?i)(?:section|part)\s+[3III]\s*[:\-–]?\s*(?:exclusions|what is not covered)", "Exclusions"),
            (r"(?i)(?:section|part)\s+[4IV]\s*[:\-–]?\s*(?:general conditions|claims procedure|portability)", "General Conditions & Portability"),
            (r"(?i)(?:section|part)\s+[5V]\s*[:\-–]?\s*(?:definitions)", "Definitions"),
            (r"(?i)^definitions", "Definitions"),
            (r"(?i)^exclusions", "Exclusions"),
            (r"(?i)^scope of cover", "Scope of Cover"),
            (r"(?i)^waiting periods?", "Waiting Periods"),
        ]

        # Key clause patterns (e.g. 1.1 Inpatient, 2.1 30-day, etc.)
        clause_pattern = re.compile(
            r"(?:^(?:[0-9]+\.[0-9]+|[A-Za-z]\.|\([0-9a-z]+\)|[0-9]+\))\s+([^\n]{3,80}))|"
            r"(?:inpatient treatment|day care|domiciliary hospitalization|pre-hospitalization|post-hospitalization|ambulance|"
            r"30\s*days? waiting period|pre-existing|specific disease waiting|portability|hospital definition|room rent|cataract|cosmetic|experimental)",
            re.MULTILINE | re.IGNORECASE
        )

        for p_info in pages:
            p_num = p_info["page_number"]
            text = p_info["text"]
            lines = text.split("\n")

            current_clause = "General Provisions"
            clause_buffer: List[str] = []

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                # Check for section transition
                for pattern, sec_name in section_patterns:
                    if re.search(pattern, line_str):
                        # Flush previous clause buffer if any
                        if clause_buffer:
                            buf_text = " ".join(clause_buffer).strip()
                            if len(buf_text) > 40:
                                chunk_id = f"POL-P{p_num:02d}-C{chunk_counter:03d}"
                                chunks.append(PolicyChunk(
                                    chunk_id=chunk_id,
                                    source=self.pdf_path.name,
                                    page=p_num,
                                    section=current_section,
                                    clause=current_clause,
                                    text=buf_text,
                                    keywords=self._extract_keywords(buf_text)
                                ))
                                chunk_counter += 1
                            clause_buffer = []
                        current_section = sec_name
                        break

                # Check for clause header
                clause_match = clause_pattern.search(line_str)
                if clause_match and len(line_str) < 100:
                    if clause_buffer:
                        buf_text = " ".join(clause_buffer).strip()
                        if len(buf_text) > 40:
                            chunk_id = f"POL-P{p_num:02d}-C{chunk_counter:03d}"
                            chunks.append(PolicyChunk(
                                chunk_id=chunk_id,
                                source=self.pdf_path.name,
                                page=p_num,
                                section=current_section,
                                clause=current_clause,
                                text=buf_text,
                                keywords=self._extract_keywords(buf_text)
                            ))
                            chunk_counter += 1
                        clause_buffer = []
                    current_clause = line_str
                else:
                    clause_buffer.append(line_str)

            # Flush remaining buffer on page
            if clause_buffer:
                buf_text = " ".join(clause_buffer).strip()
                if len(buf_text) > 40:
                    chunk_id = f"POL-P{p_num:02d}-C{chunk_counter:03d}"
                    chunks.append(PolicyChunk(
                        chunk_id=chunk_id,
                        source=self.pdf_path.name,
                        page=p_num,
                        section=current_section,
                        clause=current_clause,
                        text=buf_text,
                        keywords=self._extract_keywords(buf_text)
                    ))
                    chunk_counter += 1

        return chunks

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "shall", "under",
            "such", "will", "any", "are", "not", "have", "been", "which", "all"
        }
        unique = []
        for t in tokens:
            if t not in stopwords and t not in unique:
                unique.append(t)
        return unique[:20]

    def save_chunks_to_json(self, chunks: List[PolicyChunk], output_path: Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
