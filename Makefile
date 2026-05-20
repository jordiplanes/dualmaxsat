
all : maxsat_decomposition_report.pdf

%.pdf : %.tex
	pdflatex $<

