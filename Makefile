
PARTS = part1.tex part2.tex part3.tex part_closing.tex part_overview.tex part_taxonomy.tex abstract.tex
BIB   = refs.bib
PLOTS = benchmark_plot.png additional_plot.png scheduling_plot_1.png scheduling_plot_2.png scheduling_plot_3.png

all : memo.pdf

memo.pdf : memo.tex $(PARTS) $(BIB) $(PLOTS)
	pdflatex memo.tex
	bibtex memo
	pdflatex memo.tex
	pdflatex memo.tex

clean :
	rm -f memo.aux memo.log memo.out memo.toc memo.bbl memo.blg memo.pdf

.PHONY : all clean
