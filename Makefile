
PARTS = part1.tex part2.tex part3.tex
BIB   = refs.bib

all : memo.pdf

memo.pdf : memo.tex $(PARTS) $(BIB)
	pdflatex memo.tex
	bibtex memo
	pdflatex memo.tex
	pdflatex memo.tex

clean :
	rm -f memo.aux memo.log memo.out memo.toc memo.bbl memo.blg memo.pdf

.PHONY : all clean
