
all : maxsat_decomposition_report.pdf ihs_dantzig_wolfe_example.pdf

%.pdf : %.tex
	pdflatex $<

