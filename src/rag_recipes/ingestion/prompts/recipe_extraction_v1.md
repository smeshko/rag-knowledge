<!--
RETIRED — retained for reproducibility only.

This is the exact template that produced every `ExtractionRun` row with
`prompt_version = "recipe-extraction-v1"`. Nothing loads it: `extraction.py`
reads `_PROMPT_RESOURCE` (currently `recipe_extraction_v2.md`). It stays in the
package so those rows can be re-rendered from source (docs 11 §6 — prompt
versions are part of the reproducibility story). Do not edit.
-->
<!--
recipe-extraction prompt template, version `recipe-extraction-v1`.

CONTRACT: this template is part of the LLM API contract. Any observable change to
the instructions below — anything the model can act on differently — MUST bump
`PROMPT_VERSION` in `extraction.py` (it feeds `compute_input_hash`, which is the
extraction cache / dedup key). Editing this file without bumping the version can
silently reuse a stale cached extraction against new instructions.

The single placeholder `{source_spans}` is replaced (via str.replace, not
str.format) with the formatted page window. Do not add other `{...}` tokens.
-->

You extract recipes from the text of a cookbook or recipe document.

You are given one window of source spans taken from consecutive PDF pages. Each
span is delimited like `[SOURCE_SPAN <span_id> | PDF page <N>]` followed by its
text. The `<span_id>` values are load-bearing identifiers — you must cite the
exact ids you were given.

## Task

Identify every distinct recipe contained in this window and return it as a
`recipe.v1` item. A recipe is a self-contained set of ingredients and/or method
steps for preparing a dish. If the window contains no recipe, return an empty
`items` list.

For each recipe you find:

- Set `item_type` to `"recipe"`.
- Populate `title`, `summary`, and `body_text` from the source text.
- In `source_span_ids`, list ONLY the exact `<span_id>` values (from the
  `[SOURCE_SPAN ...]` markers above) that the recipe's text was drawn from.
  Never invent a span id. Cite the same ids per step in each step's
  `source_span_ids`.
- Fill `structured_data` with the `recipe.v1` shape: keep `schema` set to
  `"recipe.v1"`; capture `yield`, prep/cook/total times, the raw and parsed
  ingredients, and the raw and parsed steps.
- Parse each ingredient into quantity, unit, and item, providing both the raw
  and a normalized form where you can.
- Provide confidence scores in `[0, 1]` at the recipe, field, ingredient, and
  step levels, reflecting how certain you are of each value.

## Rules

- Use `null` for any field whose value is genuinely unknown or absent — do not
  guess or fabricate values, and do not use empty strings as a stand-in for
  null.
- Do not merge two separate recipes into one item, and do not split a single
  recipe across multiple items.
- Return strictly the JSON shape requested (the `recipe.v1` schema); add no
  commentary outside the JSON.

## Source spans

{source_spans}
