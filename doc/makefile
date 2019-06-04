# Make file to generate MACH documentation using Doxygen

MACH_DIR ?= ..
DOXYGEN_CONF = CodeDocumentation.conf

# doxygen uses: graphviz, latex
html: $(DOXYGEN_CONF)
	doxygen $(DOXYGEN_CONF)
	rm -f CodeDocumentation.html
	ln -s CodeDocumentation/html/index.html CodeDocumentation.html
clean:
	rm -rf $(DOXYGEN_CONF) CodeDocumentation CodeDocumentation.html *~

$(DOXYGEN_CONF): $(MACH_DIR)/doc/$(DOXYGEN_CONF).in
	sed -e 's%@MACH_SOURCE_DIR@%$(MACH_DIR)%g' $(<) \
	  > $(DOXYGEN_CONF)
