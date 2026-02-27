#!/bin/bash
# Convert all .md files in this directory to PDF via pandoc with pdflatex
# Usage: ./convert_md_to_pdf.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

for mdfile in *.md; do
    pdffile="${mdfile%.md}.pdf"
    echo "Converting: $mdfile -> $pdffile"
    pandoc "$mdfile" -o "$pdffile" \
        --pdf-engine=pdflatex \
        -V geometry:margin=0.5in
    if [ $? -eq 0 ]; then
        echo "  OK"
    else
        echo "  FAILED"
    fi
done

echo ""
echo "Done. Generated PDFs:"
ls -lh *.pdf 2>/dev/null
