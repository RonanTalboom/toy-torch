# diagrams/

Programmatic builder for the six Excalidraw diagrams that accompany each
toy-torch milestone.

The diagrams themselves live in the Obsidian vault (outside this repo) at
`…/Ronan /toy-torch diagrams/`. This folder only holds the generator.

## Regenerate

```bash
python3 diagrams/build.py
```

Outputs six `.excalidraw` JSON files + overwrites any edits made in Obsidian.

## What's drawn

| # | Milestone | References |
|---|-----------|------------|
| 01 | v0.1 core architecture | shape / tensor / op / tape / autograd / graph |
| 02 | v0.2 elementwise fusion | `src/graph/fusion.rs` |
| 03 | v0.3 codegen demo | `src/graph/codegen.rs::emit_rust` |
| 04 | v0.4 → v0.5 bench results | `benches/elementwise_chain.rs` — four-variant bar chart |
| 05 | v0.5 JIT pipeline | `src/graph/jit.rs` — emit_c → cc → dlopen |
| 06 | v0.7 reduction fusion | `src/graph/reduction.rs` — `Op::FusedSum` |

## Source-of-truth choice

Iterating on diagrams means picking one of:

1. **Edit here (this script), lose GUI edits on regen.** Good for diffable,
   version-controlled diagrams.
2. **Edit in Excalidraw GUI, stop running this script.** Good for free-form
   artistic tweaks.

If you want both — add a preamble flag to `build.py` that skips files whose
mtime is newer than a marker, and iterate in Obsidian with occasional regen
of just the specific diagram you're restructuring. Not built yet.
