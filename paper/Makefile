.PHONY: all numbers refs check clean

all: numbers check main.pdf

# Regenerate generated_numbers.tex from results/ before every build.
numbers:
	cd .. && python3 scripts/generate_numbers.py

main.pdf: main.tex sections/*.tex generated_numbers.tex refs.bib
	latexmk -pdf -interaction=nonstopmode main.tex

# Reference integrity gate -- must show 0 errors before a section is
# considered done. Run after every section, per this project's writing rule.
check:
	cd .. && python3 scripts/check_references.py --manuscript paper/sections/*.tex

clean:
	latexmk -C
	rm -f generated_numbers.tex
