# Make file to generate MISO documentation using Doxygen

MISO_DIR ?= ..
DOXYGEN_CONF = CodeDocumentation.conf

# doxygen uses: graphviz, latex
html: $(DOXYGEN_CONF)
	doxygen $(DOXYGEN_CONF)
	rm -f CodeDocumentation.html
	ln -s CodeDocumentation/html/index.html CodeDocumentation.html
clean:
	rm -rf $(DOXYGEN_CONF) CodeDocumentation CodeDocumentation.html *~

$(DOXYGEN_CONF): $(MISO_DIR)/doc/$(DOXYGEN_CONF).in
	sed -e 's%@MISO_SOURCE_DIR@%$(MISO_DIR)%g' $(<) \
	  > $(DOXYGEN_CONF)
