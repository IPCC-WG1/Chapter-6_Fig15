#!/bin/bash
template=$2
notebook=$1
echo 'Template: '$template
echo 'Notebook: '$notebook
echo $notebook'.ipynb'
ipython nbconvert $notebook'.ipynb' --to latex --template $template
latex $notebook'.tex'
bibtex $notebook'.aux'
pdflatex $notebook'.tex'
pdflatex $notebook'.tex'
pdflatex $notebook'.tex'

rm *.bbl *.aux *.blg *.log *.out *.dvi