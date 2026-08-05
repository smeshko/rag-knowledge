"""Regenerate the parsed cookbook goldens (Epic 23.2 TASK-003).

The `cookbooks` fixture set — excerpts AND goldens — is gitignored because this
repository is public (PR #61). Reproducibility is preserved by committing the
generators instead of the generated files:

  1. `data/fixtures/cookbooks_ranges.json` + `rag-evals fixtures cut`
     regenerate every `source.md` from the source PDFs.
  2. THIS script regenerates the 28 parser-derived `expected.json` files:
     `bonebrothmiracle`, `edwardiancooking`, `bakingwithlesssugar`,
     `onepantorulethemall`.

The remaining 14 goldens (`eatdrinkpaleocookbook`, `wastefreekitchenhandbook`)
were hand-drafted — their books carry no structural markers a parser can anchor
on — and exist ONLY in the local working copy. They cannot be committed in any
form (a golden's steps text is the recipe's method near-verbatim). Back up
`data/fixtures/synthetic_recipes/cookbooks/` before reformatting the machine.

Titles and yields for `bakingwithlesssugar`/`onepantorulethemall` are hand
transcriptions (the CORRECTIONS/ONEPAN_TITLES tables): running heads set in
small caps and yields split across three lines defeated every heuristic tried,
and a wrong guess would be silently scored against forever.

Usage (from backend/):
    uv run python scripts/regen_cookbook_goldens.py            # verify only
    uv run python scripts/regen_cookbook_goldens.py --write    # (re)write
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SET = Path("data/fixtures/synthetic_recipes/cookbooks")

# A digit is required after the keyword: prose like "makes for a satisfying
# flavor" sits in these headnotes and matched a laxer pattern.
_YIELD = re.compile(r"^\s*\(?((?:serves|makes)\s+(?:about\s+)?\d[^)]*?)\)?\s*$", re.I)
_UNITS = {
    "tbsp": "tablespoon", "tbsps": "tablespoon", "tablespoon": "tablespoon",
    "tablespoons": "tablespoon", "tbs.": "tablespoon", "tbs": "tablespoon",
    "tsp": "teaspoon", "tsps": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
    "cup": "cup", "cups": "cup", "lb": "pound", "lbs": "pound", "lbs.": "pound",
    "pound": "pound", "pounds": "pound", "oz": "ounce", "ounce": "ounce", "ounces": "ounce",
    "g": "gram", "grams": "gram", "kg": "kilogram", "ml": "milliliter", "l": "liter",
    "quart": "quart", "quarts": "quart", "pint": "pint", "pints": "pint",
    "clove": "clove", "cloves": "clove", "inch": "inch", "inches": "inch",
    "sprig": "sprig", "sprigs": "sprig", "package": "package", "packages": "package",
    "can": "can", "cans": "can", "stick": "stick", "sticks": "stick",
}
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
_QTY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*([½¼¾⅓⅔⅛])?\s*")


def parse_quantity(line: str) -> tuple[float | None, str]:
    """Leading quantity as a float, plus the remainder of the line."""
    m = _QTY_RE.match(line)
    if not m or (m.group(1) is None and m.group(2) is None):
        return None, line.strip()
    total = float(m.group(1)) if m.group(1) else 0.0
    if m.group(2):
        total += _FRACTIONS[m.group(2)]
    return total, line[m.end():].strip()


def parse_ingredient(raw: str) -> dict:
    """One ingredient line -> the five sub-fields the scorer compares."""
    raw = raw.strip().lstrip("•").strip()
    quantity, rest = parse_quantity(raw)

    unit = None
    words = rest.split()
    if words:
        head = words[0].lower().rstrip(".,")
        if head in _UNITS:
            unit = _UNITS[head]
            rest = " ".join(words[1:])

    # Preparation is whatever follows the first comma ("1 onion, diced").
    preparation = None
    if "," in rest:
        item, _, preparation = rest.partition(",")
        preparation = preparation.strip() or None
    else:
        item = rest
    item = item.strip().rstrip(".").lower() or None

    return {
        "raw_text": raw,
        "quantity_value": quantity,
        "unit_normalized": unit,
        "item_normalized": item,
        "preparation": preparation,
    }


def _clean(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_bonebroth(text: str) -> dict | None:
    """Layout: <running head> / <page no> / Title / headnote / Serves N /
    Ingredients: ... / Method: 1. ... — markers are literal and reliable."""
    lines = _clean(text)
    try:
        ing_at = next(i for i, ln in enumerate(lines) if ln.rstrip(":").lower() == "ingredients")
        method_at = next(i for i, ln in enumerate(lines) if ln.rstrip(":").lower() == "method")
    except StopIteration:
        return None

    yield_value = next((ln.strip() for ln in lines[:ing_at] if _YIELD.match(ln)), None)

    # The title is the FIRST substantial line after the running head and folio,
    # not the last before the yield — the headnote sits between them and its
    # closing line is not a title. The headnote opens with a dropped capital
    # extracted as its own single-character line, which bounds the search.
    title = None
    for ln in lines[:ing_at]:
        if ln.isdigit() or ln.upper() == ln or len(ln) < 4:
            continue  # running head (all caps), folio, or dropped capital
        title = ln
        break
    if not title:
        return None

    ingredients = [
        parse_ingredient(ln)
        for ln in lines[ing_at + 1 : method_at]
        # Ratio footnotes start with "*" and are guidance, not ingredients.
        if not ln.startswith("*")
    ]
    steps = []
    for ln in lines[method_at + 1 :]:
        if re.match(r"^\d+\.\s*", ln):
            steps.append(re.sub(r"^\d+\.\s*", "", ln))
        elif steps:
            steps[-1] += " " + ln
    return {"title": title, "yield": yield_value, "ingredients": ingredients, "steps": steps}


def parse_edwardian(text: str) -> dict | None:
    """Layout: Title / (serves N) / headnote / 'Ingredients needed to make X:' /
    ... / 'Steps:' / 1. ..."""
    lines = _clean(text)
    try:
        ing_at = next(
            i for i, ln in enumerate(lines) if ln.lower().startswith("ingredients needed to make")
        )
        steps_at = next(i for i, ln in enumerate(lines) if ln.rstrip(":").lower() == "steps")
    except StopIteration:
        return None

    title, yield_value = None, None
    for i, ln in enumerate(lines[:ing_at]):
        m = _YIELD.match(ln)
        if m:
            yield_value = m.group(1).strip()
            title = lines[i - 1].strip() if i else None
            break
    if not title:
        return None

    noise = re.compile(r"^(t t t|e e e|[\d|\s]+|DINNER AT THE ABBEY.*|HIGH TEA.*|DESSERTS.*)$")
    ingredients = [
        parse_ingredient(ln)
        for ln in lines[ing_at + 1 : steps_at]
        if not noise.match(ln)
    ]
    steps = []
    for ln in lines[steps_at + 1 :]:
        if noise.match(ln):
            continue
        if re.match(r"^\d+\.\s*", ln):
            steps.append(re.sub(r"^\d+\.\s*", "", ln))
        elif steps:
            steps[-1] += " " + ln
    return {"title": title, "yield": yield_value, "ingredients": ingredients, "steps": steps}




_CAPS = re.compile(r"^[A-ZÉÈÊÀÂÎÔÛÇ][A-Z0-9ÉÈÊÀÂÎÔÛÇ \-’'&,()\.]{2,}$")
_NUM_STEP = re.compile(r"^(\d+)\.\s*")


def parse_baking(text: str) -> dict | None:
    lines = _clean(text)
    bullets = [i for i, ln in enumerate(lines) if ln.startswith("•")]
    steps_at = [i for i, ln in enumerate(lines) if _NUM_STEP.match(ln)]
    if not bullets or not steps_at:
        return None

    # Title: the ALL-CAPS lines between the running head and the first bullet.
    # It wraps across lines ("PUMPKIN-WALNUT" / "CHEESECAKE BARS").
    title_parts = [
        ln
        for ln in lines[1 : bullets[0]]
        if _CAPS.match(ln) and "BAKING WITH LESS SUGAR" not in ln and "/" not in ln
    ]
    if not title_parts:
        return None
    title = " ".join(title_parts).title()

    # Yield: "MAKES" / "12" / "MUFFINS" on consecutive lines after the bullets.
    yield_value = None
    tail = lines[bullets[-1] + 1 : steps_at[0]]
    for i, ln in enumerate(tail):
        if ln.upper().startswith(("MAKES", "SERVES")):
            yield_value = " ".join(tail[i : i + 3]).lower().strip()
            break

    # Ingredients: bulleted lines, each possibly wrapped onto following lines.
    # A sub-recipe label ("WALNUT CRUST") is a group heading, not an ingredient.
    ingredients = []
    for idx, start in enumerate(bullets):
        end = bullets[idx + 1] if idx + 1 < len(bullets) else steps_at[0]
        chunk = [lines[start]]
        for ln in lines[start + 1 : end]:
            if _CAPS.match(ln) or ln.upper().startswith(("MAKES", "SERVES")) or ln.isdigit():
                break
            chunk.append(ln)
        ingredients.append(parse_ingredient(" ".join(chunk)))

    steps = []
    for i in range(steps_at[0], len(lines)):
        ln = lines[i]
        if _NUM_STEP.match(ln):
            steps.append(_NUM_STEP.sub("", ln))
        elif steps and not _CAPS.match(ln) and "/" not in ln and ln != "Continued":
            steps[-1] += " " + ln
    return {"title": title, "yield": yield_value, "ingredients": ingredients, "steps": steps}


_QTY_START = re.compile(r"^[\d½¼¾⅓⅔⅛]")
_TRAILER = re.compile(r"^(TOTAL TIME|EQUIPMENT|SERVES|PREP TIME)\s*:", re.I)
_GROUP = re.compile(r"^[A-Za-z][A-Za-z \-]{2,30}:$")
_NOISE = re.compile(r"^(•|\d+|Continued.*|[A-Z][A-Za-z\s\xa0]{6,})$")

TITLES = {
    "onepantorulethemall-p108-108": "California Black Bean Chili",
    "onepantorulethemall-p136-137": "Mediterranean Bulgur Pie",
    "onepantorulethemall-p166-166": "Shenyang Tomato and Eggs",
    "onepantorulethemall-p196-196": "Roasted Polenta",
    "onepantorulethemall-p223-224": "Almond-Crusted Apple Pie",
    "onepantorulethemall-p47-48": "Chocolate Chip Cream Cheese Coffee Cake",
    "onepantorulethemall-p61-62": "Crispy Tropical Fish Tacos",
}


def _paragraphs(lines: list[str]) -> list[str]:
    """Rejoin wrapped lines into paragraphs using the column measure.

    The measure is taken over a *rolling local window*, not the whole fixture:
    a 2-page window sets different columns on each page, and one document-wide
    maximum makes the narrower page's full-width lines all look short — which
    collapsed an 8-step recipe to 3 by ending a paragraph on nearly every line.
    """
    if not lines:
        return []
    out: list[str] = []
    buf: list[str] = []
    for i, ln in enumerate(lines):
        window = lines[max(0, i - 4) : i + 5]
        measure = max(len(w) for w in window)
        buf.append(ln.strip())
        if len(ln) < measure * 0.9:  # short line ends the paragraph
            out.append(" ".join(buf).strip())
            buf = []
    if buf:
        out.append(" ".join(buf).strip())
    return [p for p in out if len(p) > 25]


def _is_ingredient(line: str) -> bool:
    """A leading quantity is necessary but nowhere near sufficient.

    A *wrapped method line* frequently opens with a number — "1 teaspoon of oil
    into the bulgur for about 2 minutes." — and classifying those as ingredients
    both invented ingredients and swallowed the steps they belonged to.
    Sentence punctuation is the discriminator: an ingredient line is a noun
    phrase, so it neither ends in a period nor contains one mid-line.
    """
    if not _QTY_START.match(line) or len(line) >= 90:
        return False
    return not line.endswith(".") and ". " not in line


def _quote_spans(lines: list[str]) -> set[int]:
    """Indices belonging to an author attribution block.

    The block is prose closed by a line starting "—Name". Walking *backwards*
    from that terminator over every line that does not open with a quantity
    bounds it exactly — a forward heuristic on line length let the quote's own
    short first line ("I could write poems to the simple black bean. Small")
    through as an ingredient.
    """
    marked: set[int] = set()
    for i, ln in enumerate(lines):
        if not ln.startswith("—"):
            continue
        marked.add(i)
        j = i - 1
        while j >= 0 and not _QTY_START.match(lines[j]) and not _TRAILER.match(lines[j]):
            marked.add(j)
            j -= 1
    return marked


def parse(name: str, text: str) -> dict | None:
    raw = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    lines = [ln.strip() for ln in raw]
    if not any(ln.rstrip(":").upper() == "INGREDIENTS" for ln in lines):
        return None

    total_time = next(
        (ln.split(":", 1)[1].strip() for ln in lines if ln.upper().startswith("TOTAL TIME")), None
    )
    serves = next(
        (ln.split(":", 1)[1].strip() for ln in lines if ln.upper().startswith("SERVES:")), None
    )

    skip = _quote_spans(lines)
    ingredients: list[str] = []
    method_raw: list[str] = []
    for i, ln in enumerate(lines):
        if i in skip or _TRAILER.match(ln) or _GROUP.match(ln) or _NOISE.match(ln):
            continue
        if ln.rstrip(":").upper() == "INGREDIENTS" or ln == TITLES.get(name):
            continue
        if _is_ingredient(ln):
            ingredients.append(ln)
        # NOT relaxed to catch quantity-less pantry items ("Cooking spray"):
        # a `len < 30 and no period` rule recovered that one line on p47-48 and
        # invented false ingredients on five other fixtures. One known miss beats
        # five silent additions — the verifier can add it, but cannot see what
        # a loose rule invented without re-reading every source.
        elif ingredients and (ln.startswith("(") or ln[:1].islower()) and len(ln) < 60:
            ingredients[-1] += " " + ln  # wrapped ingredient, not a new one
        else:
            method_raw.append(raw[i])

    steps = _paragraphs(method_raw)
    if not ingredients or not steps:
        return None

    return {
        "item_type": "recipe",
        "title": TITLES[name],
        "source_span_ids": [f"span_eval_{name}"],
        "structured_data": {
            "schema": "recipe.v1",
            "yield": f"serves {serves}" if serves else None,
            "prep_time": None,
            "cook_time": None,
            "total_time": total_time,
            "ingredients": [parse_ingredient(i) for i in ingredients],
            "steps": [{"text": s} for s in steps],
        },
    }




CORRECTIONS = {
 "bakingwithlesssugar-p105-105": (
     "Honey-Champagne Sabayon Parfaits with Fresh Berries", "serves 6 to 8"),
 "bakingwithlesssugar-p113-113": ("Pain d’Épices", "makes one 9-in [23-cm] loaf"),
 "bakingwithlesssugar-p140-141": ("Maple Crème Caramel", "makes 8 custards"),
 "bakingwithlesssugar-p174-176": (
     "Orange Granita with Pears, Cranberries, and Citrus", "serves 6 to 8"),
 "bakingwithlesssugar-p34-36":  ("Blueberry Bran Muffins", "makes 12 muffins"),
 "bakingwithlesssugar-p44-46":  (
     "Cameron’s Lemon-Polenta-Pistachio Buttons", "makes about 12 cookies"),
 "bakingwithlesssugar-p97-97":  ("Cherry Almond Granola", "makes about 8 cups"),
 "onepantorulethemall-p108-108":  ("California Black Bean Chili", "serves 4"),
 "onepantorulethemall-p136-137":  ("Mediterranean Bulgur Pie", "serves 4–6"),
 "onepantorulethemall-p166-166":  ("Shenyang Tomato and Eggs", "serves 4–6"),
 "onepantorulethemall-p196-196":  ("Roasted Polenta", "serves 6"),
 "onepantorulethemall-p223-224":  ("Almond-Crusted Apple Pie", "serves 8–10"),
 "onepantorulethemall-p47-48":    ("Chocolate Chip Cream Cheese Coffee Cake", "serves 4–6"),
 "onepantorulethemall-p61-62":    ("Crispy Tropical Fish Tacos", "serves 4–6"),
}



def _assemble(name: str, parsed: dict) -> dict:
    return {
        "item_type": "recipe",
        "title": parsed["title"],
        "source_span_ids": [f"span_eval_{name}"],
        "structured_data": {
            "schema": "recipe.v1",
            "yield": parsed["yield"],
            "prep_time": None,
            "cook_time": None,
            "total_time": None,
            "ingredients": parsed["ingredients"],
            "steps": [{"text": s} for s in parsed["steps"]],
        },
    }


def regenerate() -> dict[str, dict]:
    """name -> golden for every parser-derived fixture present on disk."""
    goldens: dict[str, dict] = {}
    for fixture in sorted(SET.iterdir()):
        if not fixture.is_dir():
            continue
        book = fixture.name.rsplit("-p", 1)[0]
        text = (fixture / "source.md").read_text(encoding="utf-8")
        if book == "bonebrothmiracle":
            parsed = parse_bonebroth(text)
        elif book == "edwardiancooking":
            parsed = parse_edwardian(text)
        elif book == "bakingwithlesssugar":
            parsed = parse_baking(text)
            if parsed and fixture.name in CORRECTIONS:
                parsed["title"], parsed["yield"] = CORRECTIONS[fixture.name]
        elif book == "onepantorulethemall":
            golden = parse(fixture.name, text)
            if golden is None:
                raise SystemExit(f"{fixture.name}: onepan parser produced nothing")
            golden["structured_data"]["yield"] = CORRECTIONS.get(
                fixture.name, (None, golden["structured_data"]["yield"])
            )[1]
            goldens[fixture.name] = golden
            continue
        else:
            continue  # hand-drafted book — not regenerable
        if parsed is None:
            raise SystemExit(f"{fixture.name}: parser produced nothing")
        parsed["ingredients"] = [
            i if isinstance(i, dict) else parse_ingredient(i) for i in parsed["ingredients"]
        ]
        goldens[fixture.name] = _assemble(fixture.name, parsed)
    return goldens


def main() -> None:
    write = "--write" in sys.argv
    goldens = regenerate()
    mismatched = []
    for name, golden in goldens.items():
        path = SET / name / "expected.json"
        rendered = json.dumps(golden, indent=2, ensure_ascii=False) + "\n"
        if write:
            path.write_text(rendered, encoding="utf-8")
        elif not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            mismatched.append(name)
    if write:
        print(f"wrote {len(goldens)} parsed goldens")
    elif mismatched:
        raise SystemExit(f"{len(mismatched)} golden(s) differ from regeneration: {mismatched}")
    else:
        print(f"all {len(goldens)} parsed goldens verified byte-identical")


if __name__ == "__main__":
    main()
