CC = gcc

SOURCEDIR = .

EXE   = gp

SOURCES  = $(SOURCEDIR)/evaluator.c \
           $(SOURCEDIR)/generator.c \
           $(SOURCEDIR)/operation.c \
           $(SOURCEDIR)/feature_parser.c \
           $(SOURCEDIR)/main.c

IDIR      = -I.

OBJS        = $(SOURCES:.cu=.o)

CFLAGS     = -O3

# NVCCFLAGS  = -use_fast_math -O3 -std=c++11 -dc -arch=compute_61 -code=sm_61

LFLAGS      = -lm

$(EXE) : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

##$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cpp
##	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu
	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

clean:
	rm -f *.o $(EXE)
