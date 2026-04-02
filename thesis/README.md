# Thesis

Bachelor's thesis source, built with LaTeX.

## Prerequisites

- A LaTeX distribution: [TeX Live](https://tug.org/texlive/) (Linux/macOS) or [MiKTeX](https://miktex.org/) (Windows). Full installation recommended.
- `latexmk` — included in TeX Live and MiKTeX, but **requires Perl**.
  - **Windows (MiKTeX):** MiKTeX does not bundle Perl. Install [Strawberry Perl](https://strawberryperl.com/) first, then latexmk will work.
  - **Linux/macOS (TeX Live):** Perl is pre-installed; latexmk works out of the box.

## Quick Start (requires Perl / Strawberry Perl on Windows)

```bash
latexmk -cd src/thesis.tex
```

The compiled PDF is written to `thesis.pdf` in this directory.

## Manual Compile (no Perl needed)

Run from inside `src/`:

```bash
cd src
pdflatex -interaction=nonstopmode -output-directory=.. thesis.tex
bibtex ../thesis
pdflatex -interaction=nonstopmode -output-directory=.. thesis.tex
pdflatex -interaction=nonstopmode -output-directory=.. thesis.tex
```

Three passes are needed to resolve all cross-references and the table of contents.

## Clean Build Artifacts

```bash
latexmk -cd -C src/thesis.tex
```

Removes all generated files from `src/build/`. Does not delete `thesis.pdf`.

## Folder Structure

```
thesis/
├── README.md               — this file
├── thesis.pdf              — compiled output
└── src/
    ├── thesis.tex          — main LaTeX entry point
    ├── bachelor_thesis.cls — custom document class
    ├── thesis.bib          — bibliography database (BibTeX)
    ├── .latexmkrc          — latexmk configuration
    ├── preamble/           — packages, custom commands, author metadata
    ├── frontmatter/        — title page and thesis specification
    ├── chapters/           — content chapters and appendices
    ├── images/             — figures and images
    └── build/              — generated build artifacts (gitignored)
```

## Notes

- The document class `bachelor_thesis.cls` uses `biblatex` with `backend=bibtex`. `latexmk` handles the multi-pass bibtex cycle automatically.
- Editor magic comments (`%!TEX root = ../thesis.tex`) in chapter files let editors like TeXstudio, VS Code (LaTeX Workshop), and Vim (vimtex) identify the root file so you can compile from any open chapter.
