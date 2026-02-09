# functional-search (Universal Agent Skill)

A universal agent skill for searching functional specification Markdown files with hybrid relevance ranking.

This skill is designed for agent ecosystems like:
- Codex
- Claude Code
- Cursor (and other compatible agents via `skills`)

## Skill Type

`functional-search` is a **search/analysis skill** that:
- indexes `.md` files recursively in a target path,
- ranks matches with hybrid BM25-style relevance boosts,
- supports quoted phrases and `NOT` filtering,
- returns grouped results by file and section.

## Repository Layout

- `SKILL.md` — main skill instructions and invocation contract.
- `scripts/functional_search.py` — search engine script used by the skill.

## Install / Update (`npx skills`)

List available skills in this repo:

```bash
npx skills add newuni/magi-functional-search --list
```

Install globally for Codex:

```bash
npx skills add newuni/magi-functional-search --skill functional-search --agent codex -g -y
```

Install globally for Claude Code:

```bash
npx skills add newuni/magi-functional-search --skill functional-search --agent claude-code -g -y
```

Install for all supported agents:

```bash
npx skills add newuni/magi-functional-search --skill functional-search --agent '*' -g -y
```

Update / re-sync:

```bash
npx skills add newuni/magi-functional-search --skill functional-search --agent '*' -g -y
```

Force clean reinstall (optional):

```bash
npx skills remove functional-search --agent '*' -g -y
npx skills add newuni/magi-functional-search --skill functional-search --agent '*' -g -y
```

## Quick Usage

Natural search:

```text
well content filters
```

Exact phrase boost:

```text
"well content"
```

Exclude terms:

```text
compound NOT image
```

With path:

```text
filter wells by compound in ./docs
```

Optional flags:
- `--toc`
- `--section "<name>"`
- `--stats`
- `--format json`
- `--max-results N`
- `--context N`
