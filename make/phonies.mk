#phony variables to make various things
.PHONY : dats
dats : $(DAT_FILES)

.PHONY : jpegs
jpegs : $(JPG_FILES)

.PHONY : clean
clean :
	rm $(DAT_FILES) $(JPG_FILES) analysis.tar.gz

.PHONY : all
all :
	make dats && make jpegs && make analysis.tar.gz
