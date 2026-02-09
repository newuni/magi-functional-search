#!/usr/bin/env python3
"""Search functional specification markdown files with a single hybrid engine.

One search that combines BM25 ranking, exact phrase boost, section-title boost,
proximity boost, and NOT filtering. No modes to choose — just type your query.
"""

import argparse
import difflib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Tokenisation & stop words
# ---------------------------------------------------------------------------

TOKENIZE_RE = re.compile(r"[^\w]+", re.UNICODE)

STOP_WORDS = frozenset({
    # Spanish
    "a", "al", "con", "de", "del", "el", "en", "es", "la", "las", "lo",
    "los", "no", "o", "para", "por", "que", "se", "su", "un", "una", "y",
    "como", "más", "pero", "sin", "sobre", "entre", "hasta", "desde",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "ser", "está", "son", "fue", "sido", "tiene", "tiene",
    # English
    "the", "an", "and", "or", "of", "in", "to", "is", "it", "for",
    "on", "with", "as", "at", "by", "this", "that", "from", "be", "are",
    "was", "were", "been", "has", "have", "had", "will", "would", "can",
    "could", "should", "may", "might", "do", "does", "did", "not",
})


def tokenize(text: str) -> list[str]:
    return [t for t in TOKENIZE_RE.split(text.lower()) if t and t not in STOP_WORDS]


# ---------------------------------------------------------------------------
# Header regex
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(
    r"^[\d.\s]*"
    r"(#{1,6})\s+"
    r"(.+?)"
    r"(?:\s*\{#[^}]*\})?"
    r"\s*$"
)


# ---------------------------------------------------------------------------
# Section tree
# ---------------------------------------------------------------------------

@dataclass
class SectionNode:
    title: str
    level: int
    start_line: int
    end_line: int = 0
    children: list["SectionNode"] = field(default_factory=list)
    parent: Optional["SectionNode"] = None

    def breadcrumb(self) -> str:
        parts = []
        node = self
        while node:
            parts.append(node.title)
            node = node.parent
        return " > ".join(reversed(parts))


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

@dataclass
class MarkdownDocument:
    path: Path
    lines: list[str] = field(default_factory=list)
    sections: list[SectionNode] = field(default_factory=list)
    _all_sections: list[SectionNode] = field(default_factory=list)

    @classmethod
    def parse(cls, path: Path) -> "MarkdownDocument":
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        doc = cls(path=path, lines=lines)
        doc._parse_sections()
        return doc

    def _parse_sections(self) -> None:
        flat: list[SectionNode] = []
        for i, line in enumerate(self.lines):
            m = HEADER_RE.match(line)
            if not m:
                continue
            level = len(m.group(1))
            title = m.group(2).strip()
            title = re.sub(r"[*_`]+", "", title).strip()
            if not title:
                continue
            flat.append(SectionNode(title=title, level=level, start_line=i + 1))

        for idx, node in enumerate(flat):
            node.end_line = flat[idx + 1].start_line - 1 if idx + 1 < len(flat) else len(self.lines)

        stack: list[SectionNode] = []
        roots: list[SectionNode] = []
        for node in flat:
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                node.parent = stack[-1]
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)

        self.sections = roots
        self._all_sections = flat

    def toc(self) -> str:
        out = [f"Table of Contents: {self.path.name}", "=" * 60]
        for node in self._all_sections:
            out.append(f"{'  ' * (node.level - 1)}{node.title}  (line {node.start_line})")
        return "\n".join(out)

    def stats(self) -> dict:
        text = "\n".join(self.lines)
        table_rows = sum(1 for l in self.lines if l.strip().startswith("|"))
        tables, in_t = 0, False
        for l in self.lines:
            if l.strip().startswith("|"):
                if not in_t:
                    tables += 1
                    in_t = True
            else:
                in_t = False
        return {
            "file": str(self.path),
            "lines": len(self.lines),
            "words": len(text.split()),
            "sections": len(self._all_sections),
            "tables": tables,
            "table_rows": table_rows,
            "images": len(re.findall(r"!\[.*?\][\[(]", text)),
            "links": len(re.findall(r"\[.*?\]\(.*?\)", text)),
        }

    def find_section(self, query: str) -> Optional[SectionNode]:
        best_score, best_node, q = 0.0, None, query.lower()
        for node in self._all_sections:
            t = node.title.lower()
            score = (0.95 + len(q) / len(t) * 0.05) if q in t else difflib.SequenceMatcher(None, q, t).ratio()
            if score > best_score:
                best_score, best_node = score, node
        return best_node if best_node and best_score >= 0.4 else None

    def section_for_line(self, lineno: int) -> Optional[SectionNode]:
        best = None
        for node in self._all_sections:
            if node.start_line <= lineno <= node.end_line:
                if best is None or node.start_line > best.start_line:
                    best = node
        return best


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class SearchMatch:
    file: str
    line_number: int
    line_text: str
    breadcrumb: str
    score: float = 0.0
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "file": self.file,
            "line": self.line_number,
            "text": self.line_text,
            "section": self.breadcrumb,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }
        if self.score > 0:
            d["score"] = round(self.score, 4)
        return d


# ---------------------------------------------------------------------------
# Query parser  — understands NOT, "phrases", and bare terms
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    positive_phrases: list[str]       # "exact phrase" → must appear as-is
    positive_tokens: list[str]        # individual words for BM25
    excluded_tokens: list[str]        # words after NOT
    excluded_phrases: list[str]       # "phrase" after NOT

    @classmethod
    def parse(cls, raw: str) -> "ParsedQuery":
        pos_phrases, neg_phrases = [], []
        pos_tokens, neg_tokens = [], []
        # Pull out "quoted phrases" first
        phrases = re.findall(r'"([^"]+)"', raw)
        remaining = re.sub(r'"[^"]+"', " ", raw)
        # Split remaining on NOT (case-sensitive to avoid Spanish "not")
        parts = re.split(r'\bNOT\b', remaining)
        positive_part = parts[0]
        negative_part = " ".join(parts[1:]) if len(parts) > 1 else ""

        # Determine which phrases are positive/negative based on position
        # Simple heuristic: phrases before NOT are positive
        not_pos = remaining.upper().find(" NOT ")
        for ph in phrases:
            ph_pos = raw.find(f'"{ph}"')
            if not_pos >= 0 and ph_pos > not_pos:
                neg_phrases.append(ph.lower())
            else:
                pos_phrases.append(ph.lower())

        pos_tokens = tokenize(positive_part)
        neg_tokens = tokenize(negative_part)

        return cls(
            positive_phrases=pos_phrases,
            positive_tokens=pos_tokens,
            excluded_tokens=neg_tokens,
            excluded_phrases=neg_phrases,
        )

    @property
    def all_positive_terms(self) -> list[str]:
        """All positive tokens including phrase words, for BM25."""
        extra = []
        for ph in self.positive_phrases:
            extra.extend(tokenize(ph))
        return list(dict.fromkeys(self.positive_tokens + extra))  # dedup, preserve order


# ---------------------------------------------------------------------------
# Hybrid search engine
# ---------------------------------------------------------------------------

class HybridSearch:
    """Single search combining multiple signals into one score per line.

    Signals:
      1. BM25 (line-level)        — relevance of individual line
      2. BM25 (section-level)     — relevance of containing section (propagated)
      3. Exact phrase boost        — query substring found verbatim
      4. Section title boost       — query terms match a section header
      5. Proximity boost           — query terms co-occur within N lines
      6. NOT filtering             — lines containing excluded terms are dropped
    """

    # Weights for combining signals (tunable)
    W_BM25_LINE = 1.0
    W_BM25_SECTION = 0.4
    W_EXACT_PHRASE = 3.0
    W_SECTION_TITLE = 2.0
    W_PROXIMITY = 1.5
    PROXIMITY_WINDOW = 5  # lines

    def __init__(self, docs: list[MarkdownDocument], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b

        # Build line-level index
        self.line_entries: list[tuple[MarkdownDocument, int, list[str]]] = []
        self.line_df: Counter = Counter()
        for doc in docs:
            for i, line in enumerate(doc.lines):
                tokens = tokenize(line)
                if tokens:
                    self.line_entries.append((doc, i, tokens))
                    for t in set(tokens):
                        self.line_df[t] += 1
        self.n_lines = len(self.line_entries)
        self.avg_line_dl = (sum(len(e[2]) for e in self.line_entries) / self.n_lines) if self.n_lines else 1

        # Build section-level index
        self.section_entries: list[tuple[MarkdownDocument, SectionNode, list[str]]] = []
        self.section_df: Counter = Counter()
        for doc in docs:
            for section in doc._all_sections:
                text = " ".join(doc.lines[section.start_line - 1:section.end_line])
                tokens = tokenize(text)
                if tokens:
                    self.section_entries.append((doc, section, tokens))
                    for t in set(tokens):
                        self.section_df[t] += 1
        self.n_sections = len(self.section_entries)
        self.avg_sec_dl = (sum(len(e[2]) for e in self.section_entries) / self.n_sections) if self.n_sections else 1

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str],
                    df: Counter, n: int, avg_dl: float) -> float:
        tf_map = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            d = df.get(qt, 0)
            idf = math.log((n - d + 0.5) / (d + 0.5) + 1.0)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / max(avg_dl, 1)))
        return score

    def search(self, raw_query: str, context: int = 3,
               max_results: int = 20) -> list[SearchMatch]:
        pq = ParsedQuery.parse(raw_query)
        q_tokens = pq.all_positive_terms
        if not q_tokens and not pq.positive_phrases:
            return []

        # --- Pre-compute section BM25 scores ---
        # Map (doc_path, section_start) → section BM25 score
        section_scores: dict[tuple[str, int], float] = {}
        for doc, section, tokens in self.section_entries:
            s = self._bm25_score(q_tokens, tokens, self.section_df, self.n_sections, self.avg_sec_dl)
            if s > 0:
                section_scores[(str(doc.path), section.start_line)] = s

        # --- Pre-compute per-line token positions for proximity ---
        # For each doc, map token → set of line indices
        doc_token_lines: dict[str, dict[str, set[int]]] = {}
        for doc in self.docs:
            tl: dict[str, set[int]] = {}
            for i, line in enumerate(doc.lines):
                for t in tokenize(line):
                    tl.setdefault(t, set()).add(i)
            doc_token_lines[str(doc.path)] = tl

        # --- Score each line ---
        scored: dict[tuple[str, int], tuple[float, MarkdownDocument, int]] = {}

        for doc, line_idx, tokens in self.line_entries:
            line = doc.lines[line_idx]
            line_lower = line.lower()
            doc_key = str(doc.path)

            # --- NOT filter: skip lines containing excluded terms ---
            if pq.excluded_tokens:
                line_token_set = set(tokens)
                if any(et in line_token_set for et in pq.excluded_tokens):
                    continue
            if pq.excluded_phrases:
                if any(ep in line_lower for ep in pq.excluded_phrases):
                    continue

            score = 0.0

            # Signal 1: BM25 line-level
            bm25_line = self._bm25_score(q_tokens, tokens, self.line_df, self.n_lines, self.avg_line_dl)
            if bm25_line == 0 and not pq.positive_phrases:
                continue  # No relevance at all
            score += self.W_BM25_LINE * bm25_line

            # Signal 2: BM25 section-level (propagate section score to its lines)
            section = doc.section_for_line(line_idx + 1)
            if section:
                sec_key = (doc_key, section.start_line)
                sec_score = section_scores.get(sec_key, 0.0)
                score += self.W_BM25_SECTION * sec_score

            # Signal 3: Exact phrase boost
            for phrase in pq.positive_phrases:
                if phrase in line_lower:
                    score += self.W_EXACT_PHRASE
            # Also boost full query as substring (when no explicit phrases)
            if not pq.positive_phrases:
                full_q = raw_query.lower().strip()
                if len(full_q) > 3 and full_q in line_lower:
                    score += self.W_EXACT_PHRASE * 0.5

            # Signal 4: Section title boost
            if section:
                title_tokens = set(tokenize(section.breadcrumb()))
                overlap = len(set(q_tokens) & title_tokens)
                if overlap:
                    score += self.W_SECTION_TITLE * (overlap / max(len(q_tokens), 1))

            # Signal 5: Proximity boost — are multiple query terms near this line?
            if len(q_tokens) >= 2:
                tl = doc_token_lines.get(doc_key, {})
                nearby_count = 0
                for qt in q_tokens:
                    qt_lines = tl.get(qt, set())
                    if any(abs(line_idx - ol) <= self.PROXIMITY_WINDOW for ol in qt_lines):
                        nearby_count += 1
                if nearby_count >= 2:
                    proximity_ratio = nearby_count / len(q_tokens)
                    score += self.W_PROXIMITY * proximity_ratio

            if score > 0:
                key = (doc_key, line_idx)
                if key not in scored or scored[key][0] < score:
                    scored[key] = (score, doc, line_idx)

        # Also catch lines that match phrases but had no BM25 hit (e.g., all stop words)
        if pq.positive_phrases:
            for doc in self.docs:
                doc_key = str(doc.path)
                for i, line in enumerate(doc.lines):
                    key = (doc_key, i)
                    if key in scored:
                        continue
                    line_lower = line.lower()
                    phrase_score = 0.0
                    for phrase in pq.positive_phrases:
                        if phrase in line_lower:
                            phrase_score += self.W_EXACT_PHRASE
                    if phrase_score > 0:
                        # Check NOT filter
                        if pq.excluded_tokens:
                            lt = set(tokenize(line))
                            if any(et in lt for et in pq.excluded_tokens):
                                continue
                        if pq.excluded_phrases and any(ep in line_lower for ep in pq.excluded_phrases):
                            continue
                        scored[key] = (phrase_score, doc, i)

        # --- Sort by score and build results ---
        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)

        results = []
        for score, doc, line_idx in ranked[:max_results]:
            lineno = line_idx + 1
            section = doc.section_for_line(lineno)
            breadcrumb = section.breadcrumb() if section else "(top-level)"
            start = max(0, line_idx - context)
            end = min(len(doc.lines), line_idx + context + 1)
            results.append(SearchMatch(
                file=str(doc.path),
                line_number=lineno,
                line_text=doc.lines[line_idx],
                breadcrumb=breadcrumb,
                score=score,
                context_before=doc.lines[start:line_idx],
                context_after=doc.lines[line_idx + 1:end],
            ))
        return results


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class ResultFormatter:

    @staticmethod
    def format_text(results: list[SearchMatch], max_results: int = 20) -> str:
        if not results:
            return "No matches found."
        total = len(results)
        shown = results[:max_results]
        lines = [
            f"Found {total} match(es)" +
            (f" (showing top {max_results})" if total > max_results else "") +
            " [ranked by relevance]",
            "",
        ]
        by_file: dict[str, list[SearchMatch]] = {}
        for m in shown:
            by_file.setdefault(m.file, []).append(m)

        for filepath, matches in by_file.items():
            lines.append(f"--- {filepath} ---")
            by_section: dict[str, list[SearchMatch]] = {}
            for m in matches:
                by_section.setdefault(m.breadcrumb, []).append(m)

            for breadcrumb, section_matches in by_section.items():
                lines.append(f"  [{breadcrumb}]")
                for m in section_matches:
                    for cl in m.context_before:
                        lines.append(f"    {cl}")
                    lines.append(f"  > L{m.line_number} ({m.score:.2f}): {m.line_text}")
                    for cl in m.context_after:
                        lines.append(f"    {cl}")
                    lines.append("")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def format_json(results: list[SearchMatch], max_results: int = 20) -> str:
        total = len(results)
        shown = results[:max_results]
        return json.dumps({
            "total": total,
            "shown": len(shown),
            "matches": [m.to_dict() for m in shown],
        }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_docs(path_arg: Optional[str]) -> list[MarkdownDocument]:
    if path_arg is None:
        path_arg = "."
    p = Path(path_arg)
    if p.is_file():
        return [MarkdownDocument.parse(p)]
    if p.is_dir():
        mds = sorted(p.rglob("*.md"))
        if not mds:
            print(f"No .md files found in {p}", file=sys.stderr)
            sys.exit(1)
        return [MarkdownDocument.parse(f) for f in mds]
    matches = sorted(Path(".").glob(path_arg))
    if not matches:
        print(f"No files matching: {path_arg}", file=sys.stderr)
        sys.exit(1)
    return [MarkdownDocument.parse(f) for f in matches if f.suffix == ".md"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart search over functional spec markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Just type what you're looking for. The engine handles the rest.

examples:
  %(prog)s "filtros contenido" --path "Propuesta Ideable - ImageIQ.md"
  %(prog)s "compound concentration" --path nuevos_resultados_Api/
  %(prog)s '"well content" NOT image' --path spec.md
  %(prog)s "cómo se filtran los wells por compuesto" --path spec.md

structural commands:
  %(prog)s --toc --path "Propuesta Ideable - ImageIQ.md"
  %(prog)s --section "Finder" --path "Propuesta Ideable - ImageIQ.md"
  %(prog)s --stats --path "Propuesta Ideable - ImageIQ.md"
""",
    )
    parser.add_argument("query", nargs="?", help="Search query (natural language, phrases, NOT)")
    parser.add_argument("--path", "-p", help="File or directory (default: .)")
    parser.add_argument("--toc", action="store_true", help="Show table of contents")
    parser.add_argument("--section", "-s", help="Extract full section by name (fuzzy)")
    parser.add_argument("--stats", action="store_true", help="Document statistics")
    parser.add_argument("--context", "-c", type=int, default=3, help="Context lines (default: 3)")
    parser.add_argument("--max-results", "-n", type=int, default=20, help="Max results (default: 20)")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    if not args.toc and not args.section and not args.stats and not args.query:
        parser.error("Provide a query, or use --toc / --section / --stats")

    docs = discover_docs(args.path)

    if args.toc:
        for doc in docs:
            print(doc.toc())
            print()
        return

    if args.stats:
        for doc in docs:
            s = doc.stats()
            if args.format == "json":
                print(json.dumps(s, ensure_ascii=False, indent=2))
            else:
                print(f"Stats: {s['file']}")
                for k in ("lines", "words", "sections", "tables", "table_rows", "images", "links"):
                    print(f"  {k:12s}: {s[k]}")
                print()
        return

    if args.section:
        for doc in docs:
            node = doc.find_section(args.section)
            if node:
                print(f"--- {doc.path.name} ---")
                print(f"Section: {node.breadcrumb()}  (lines {node.start_line}-{node.end_line})")
                print("=" * 60)
                for line in doc.lines[node.start_line - 1:node.end_line]:
                    print(line)
                print()
        return

    engine = HybridSearch(docs)
    results = engine.search(args.query, context=args.context, max_results=args.max_results)

    if args.format == "json":
        print(ResultFormatter.format_json(results, args.max_results))
    else:
        print(ResultFormatter.format_text(results, args.max_results))


if __name__ == "__main__":
    main()
