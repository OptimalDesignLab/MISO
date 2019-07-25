
# WARNING: You must have the MFEM_DIR environment variable set for this to work.
# A common (perhaps not best) way to do this is by adding the following line to
# your .bashrc file:
# export MFEM_DIR=(path to mfem)/mfem

define MACH_HELP_MSG

MACH makefile targets:

  make
  make libmach
  make clean

endef

# This follows the approach in the mfem examples makefile
MFEM_BUILD_DIR ?= $(MFEM_DIR)
MFEM_INSTALL_DIR ?= $(MFEM_INSTALL_DIR)
ifdef MFEM_USE_PUMI
	CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk
else 
	CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
endif

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)  # this includes mfem's config.mk makefile

# directories
SRC_DIR=$(CURDIR)/src
TEST_DIR=$(CURDIR)/test
SANDBOX_DIR=$(CURDIR)/sandbox

# The object file dependencies are generated using the approach described at
# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/

# The following creates a hidden directory to store the dependency files
DEP_DIR=$(CURDIR)/.d
$(shell mkdir -p $(DEP_DIR) >/dev/null)
# Flags used on gcc to generate the (temporary) dependency files
DEP_FLAGS = -MT $(@) -MMD -MP -MF $(DEP_DIR)/$(*F).Td
# When executed, moves the tempary dependecies and touch the object file
POST_COMPILE = @mv -f $(DEP_DIR)/$(*F).Td $(DEP_DIR)/$(*F).d && touch $@

# set C++ compiler and flags
CXX = $(MFEM_CXX)
MACH_FLAGS = $(MFEM_FLAGS) -I$(SRC_DIR) -fPIC

# libraries needed for compiling/linking
MACH_LIBS = -ladept $(MFEM_LIBS)

# export the variables so the Make done in ./test and ./sandbox can use them
export

# source and object file names
# HEADERS = $(wildcard $(SRC_DIR)/*.hpp)
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJS= $(SOURCES:.cpp=.o)

# implicit rule for *.cpp files
%.o : %.cpp
.SECONDEXPANSION: 
%.o : %.cpp $(DEP_DIR)/$$*.d $(MFEM_LIB_FILE) $(CONFIG_MK)
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(DEP_FLAGS) $(MACH_FLAGS) -o $(@) -c $(<)
	@$(POST_COMPILE)

default: all

all: libmach.a

.PHONY: temp
temp:
	@echo $(DEP_DIR)

#solver : src/solver.cpp $(DEP_DIR)/solver.d $(MFEM_LIB_FILE) $(CONFIG_MK)
#	@echo "Compiling \""$@"\" from \""$<"\""
#	@$(CXX) $(MACH_FLAGS) -o src/solver.o -c $(<)
#	@$(POST_COMPILE)

# Create a pattern rule with an empty recipe, so that make won't fail if the
# dependency file doesn’t exist.
$(DEP_DIR)/%.d: ;
# Mark the dependency files precious to make, so they won’t be automatically
# deleted as intermediate files.
.PRECIOUS: $(DEP_DIR)/%.d

# rule for static Mach library
# TODO: if any of the *.cpp or *.hpp files change, this should recompile, but it does not
libmach.a: $(OBJS) makefile
	@echo "Compiling static Mach library"
	@ar rcs $@ $(OBJS)

.PHONY: sandbox # needed because sandbox is also the name of the directory
sandbox: libmach.a 
	@cd $(SANDBOX_DIR) && $(MAKE)

tests: libmach.a
	@cd $(TEST_DIR) && $(MAKE)

#@$(CXX) $(MACH_FLAGS) -static -Wl,-soname,libmach.so -o libmach.so $(OBJS) $(MACH_LIBS)

clean:
	@echo "deleting temporary, object, and binary files"
	@rm -f $(OBJS)
	@rm -f *~ libmach.*
	@rm -f $(DEP_DIR)/*.d
	@rmdir $(DEP_DIR)
	@cd $(SANDBOX_DIR) && $(MAKE) clean
	@cd $(TEST_DIR) && $(MAKE) clean

clean_sandbox:
	@echo "deleting temporary, object, and binary files from sandbox directory"
	@cd $(SANDBOX_DIR) && $(MAKE) clean

# Include the dependency files that exist: translate each file listed in SOURCES
# into its dependency file. Use wildcard to avoid failing on non-existent files.
include $(wildcard $(patsubst %,$(DEP_DIR)/%.d,$(notdir $(SOURCES))))
