# Use pdflatex as the PDF engine
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Output PDF to the parent directory (thesis root)
$out_dir = '..';

# Keep build artifacts (aux, log, toc, bbl, etc.) in a local build folder
$aux_dir = 'build';
