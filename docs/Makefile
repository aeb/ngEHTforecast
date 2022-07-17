# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = 
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ngEHTforecast
SOURCEDIR     = ../
BUILDDIR      = ../../ngEHTforecast-docs
PDFBUILDDIR = /tmp
PDF = ../manual.pdf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	rm -r $(BUILDDIR)/html/*
	rm -r $(BUILDDIR)/doctrees/*

htmlpush: html
	touch .docdate; cd $(BUILDDIR)/html; git add . ; git commit -m "rebuilt docs"; git push origin gh-pages

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PDFBUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(PDFBUILDDIR)/latex all-pdf
	cp $(PDFBUILDDIR)/latex/*.pdf $(PDF)
	@echo "pdflatex finished; see $(PDF)"


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

%: Makefile
	cd scripts; ./write_examples_docs.py; cp examples.rst.autogen ../src/examples.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
