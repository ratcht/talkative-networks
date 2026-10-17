#!/usr/bin/env bash
# Render a study protocol to PDF with pandoc + xelatex.
#
#   bash scripts/study_to_pdf.sh docs/studies/02-spatial-ports.md [out.pdf]
#
# Two things need handling that a bare pandoc call gets wrong.
#
# Underscores inside \texttt{} in math. The documents are authored against a
# Markdown previewer whose MathJax accepts \texttt{a_b} and renders the
# backslashes literally if the underscore is escaped. LaTeX wants the opposite:
# an unescaped _ in that position is a subscript, or in text mode an outright
# error ("Missing $ inserted"). Rather than make the source worse to read in
# the previewer it is actually edited in, the escaping is normalised here, on a
# copy. Keep both renderers happy by moving identifiers out of math mode
# instead, and this step becomes a no-op.
#
# Fonts. The document uses §, ×, —, ✓, → and ·, so the body font has to cover
# them; pandoc's default pdflatex path fails on the first §. The fonts below
# are chosen because they are present on this machine and cover all six —
# verify with `fc-list ':charset=2713' family` before substituting others.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-docs/studies/02-spatial-ports.md}"
OUT="${2:-${SRC%.md}.pdf}"
[[ -f "$SRC" ]] || { echo "no such file: $SRC" >&2; exit 1; }

command -v xelatex >/dev/null || {
  echo "xelatex not found. install with:" >&2
  echo "  sudo apt install texlive-xetex texlive-fonts-recommended texlive-latex-recommended" >&2
  exit 1
}

MAIN_FONT="${STUDY_PDF_MAIN_FONT:-DejaVu LGC Sans}"
MONO_FONT="${STUDY_PDF_MONO_FONT:-JetBrains Mono}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
norm="$tmp/$(basename "$SRC")"

python3 - "$SRC" "$norm" <<'PY'
import re, sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

def escape(match):
  body = match.group(1).replace("\\_", "_")  # normalise, then escape once
  return "\\texttt{" + body.replace("_", "\\_") + "}"

text, n = re.subn(r"\\texttt\{([^}]*)\}", escape, text)
open(dst, "w", encoding="utf-8").write(text)
print(f"[pdf] normalised {n} \\texttt{{}} identifier(s) for LaTeX")
PY

# no --number-sections: the documents number their own sections, and pandoc's
# numbering would stack on top of that ("2.1 1. Question & hypothesis")
# --resource-path because the normalised copy lives in a temp directory: a
# relative image path in the source would otherwise resolve against neither the
# copy's directory nor the repo root
pandoc "$norm" -o "$OUT" \
  --pdf-engine=xelatex \
  --resource-path="$(dirname "$SRC"):$PWD" \
  --toc --toc-depth=3 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V colorlinks=true -V linkcolor=black -V urlcolor=blue \
  -V mainfont="$MAIN_FONT" \
  -V monofont="$MONO_FONT"

echo "[pdf] wrote $OUT ($(du -h "$OUT" | cut -f1))"
