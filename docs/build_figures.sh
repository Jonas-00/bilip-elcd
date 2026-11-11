#!/usr/bin/env bash
# Convert all tikz files in figures/ to SVG via pdflatex + pdf2svg
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIG_DIR="$SCRIPT_DIR/figures"
OUT_DIR="$SCRIPT_DIR/figures"
PDFLATEX=/Library/TeX/texbin/pdflatex
PDF2SVG=/opt/homebrew/bin/pdf2svg
TMP=$(mktemp -d)

for tikz in "$FIG_DIR"/*.tikz; do
    name=$(basename "$tikz" .tikz)
    echo "→ $name"

    cat > "$TMP/$name.tex" <<TEXEOF
\documentclass[tikz,border=4pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{xcolor}
\begin{document}
\input{$tikz}
\end{document}
TEXEOF

    $PDFLATEX -interaction=nonstopmode -output-directory "$TMP" "$TMP/$name.tex" > /dev/null || true
    $PDF2SVG "$TMP/$name.pdf" "$OUT_DIR/$name.svg"
    echo "   saved $OUT_DIR/$name.svg"
done

rm -rf "$TMP"
echo "Done."
