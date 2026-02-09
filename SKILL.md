---
name: functional-search
description: Smart search over functional spec markdown files with hybrid BM25 ranking
allowed-tools: Bash(python3 *)
argument-hint: <query> [en <path>]
---

Search functional specification markdown files with hybrid ranked search.

Parse $ARGUMENTS to extract the query, optional path, and flags.

If the arguments contain ` en ` or ` in `, split on the last occurrence to get query and path. Strip surrounding quotes from path.

Run the search (portable script resolution, do not hardcode a single home path):
```bash
SCRIPT_PATH=""
for p in \
  "./scripts/functional_search.py" \
  "$HOME/.agents/skills/functional-search/scripts/functional_search.py" \
  "$HOME/.codex/skills/functional-search/scripts/functional_search.py" \
  "$HOME/.claude/skills/functional-search/scripts/functional_search.py"
do
  [ -f "$p" ] && SCRIPT_PATH="$p" && break
done

if [ -z "$SCRIPT_PATH" ]; then
  echo "functional-search script not found in local repo or known skill install paths" >&2
  exit 1
fi

# If path was provided:
python3 "$SCRIPT_PATH" "<query>" --path "<path>" [flags]

# If path was not provided:
python3 "$SCRIPT_PATH" "<query>" [flags]
```

The engine automatically combines BM25 ranking, exact phrase boost, section-title boost, proximity boost, and NOT filtering. No modes needed.

Query syntax (all automatic):
- Bare words: `filtros contenido wells` — ranked by relevance
- "Quoted phrases": `"well content"` — exact match boosted
- NOT: `compound NOT solvent` — excludes lines with those terms
- Natural language: `cómo se filtran los wells por compuesto` — just works
- Combined: `"well content" compound NOT image` — all features together

Structural commands:
- `--toc` → add `--toc` flag
- `--section "<name>"` → add `--section "<name>"` flag
- `--stats` → add `--stats` flag

Use `--format json` for programmatic output, `--max-results N` to limit, `--context N` for surrounding lines.

Default path: current working directory (all .md files recursively).

Present results grouped by file > section breadcrumb, with scores. For large result sets, summarize which sections matched first.
