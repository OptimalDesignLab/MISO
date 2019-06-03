# makefile for Mach sandbox binaries 

# source and object file names
HEADERS_SB = $(wildcard ./*.hpp)
SOURCES_SB = $(wildcard ./*.cpp)
OBJS_SB = $(SOURCES_SB:.cpp=.o)
BIN_SB = $(SOURCES_SB:.cpp=.bin)

default: all

all: $(BIN_SB)

# implicit rules

# NOTE: the object files are intermediate in the chain from .cpp to .bin, so
# Make deletes them at the end
%.o : %.cpp 
%.o : %.cpp makefile
	@echo "Compiling \""$@"\" from \""$<"\""
	@$(CXX) $(MACH_FLAGS) -o $(@) -c $(<) 

%.bin : %.o ../libmach.a 
	@echo "Compiling \""$@"\""
	@$(CXX) $(MACH_FLAGS) $< -o $@ ../libmach.a $(MACH_LIBS)

clean:
	@echo "deleting temporary, object, and binary files"
	@rm -f $(BIN_SB)