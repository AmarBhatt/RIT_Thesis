@cls
@if exist Thesis.pdf del Thesis.pdf
@bibtex Thesis
@bibtex Thesis.gls
@texify -p Thesis.tex
@if exist Thesis.pdf start Thesis.pdf
