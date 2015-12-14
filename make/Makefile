include config.mk
include wilds.mk
#Count words.
%.dat : ./books/%.txt $(COUNT_SRC)
	$(COUNT_EXE) $< $@

#isles.dat : ./books/isles.txt wordcount.py
#	python2.7 wordcount.py ./books/isles.txt isles.dat

#abyss.dat : ./books/abyss.txt wordcount.py
#	python2.7 wordcount.py ./books/abyss.txt abyss.dat

#last.dat : ./books/last.txt wordcount.py
#	python2.7 wordcount.py ./books/last.txt last.dat

%.jpg : ./%.dat $(PLOT_SRC) 
	$(PLOT_EXE) $< $@

analysis.tar.gz : $(DAT_FILES) $(JPG_FILES) $(COUNT_SRC)
	tar -czf $@ $^

include phonies.mk
