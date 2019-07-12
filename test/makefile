# makefile for Mach tests 

# source and object file names
HEADERS_TEST = $(wildcard ./*.hpp)
SOURCES_TEST = $(wildcard ./*.cpp)
OBJS_TEST = $(SOURCES_TEST:.cpp=.o)
TEST_BIN = test.bin

all: $(TEST_BIN)

default: all

# implicit rules

# NOTE: the object files are intermediate in the chain from .cpp to .bin, so
# Make deletes them at the end
%.o : %.cpp 
%.o : %.cpp makefile
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(MACH_FLAGS) -o $(@) -c $(<) 

$(TEST_BIN) : $(OBJS_TEST) ../libmach.a 
	@echo "Compiling \""$@"\""
	$(CXX) $(MACH_FLAGS) $(OBJS_TEST) -o $@ ../libmach.a $(MACH_LIBS)

clean:
	@echo "deleting temporary, object, and binary files"
	@rm -f $(TEST_BIN)