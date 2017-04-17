CC = nvcc

SOURCEDIR = .

EXE   = gp

SOURCES  = $(SOURCEDIR)/evaluator.c \
	   $(SOURCEDIR)/operation.c \
           $(SOURCEDIR)/population.c \
	   $(SOURCEDIR)/operation.c \
           $(SOURCEDIR)/generator.c \
           $(SOURCEDIR)/feature_parser.c \
           $(SOURCEDIR)/utils.c \
           $(SOURCEDIR)/kernel.cu \
           $(SOURCEDIR)/main.c

IDIR      = -I.

OBJS        = $(SOURCES:.cu=.o)

CFLAGS     = -O3

NVCCFLAGS  = -O3 -dc -arch=compute_61 -code=sm_61 -use_fast_math

LFLAGS      = -lm -lpthread -arch=compute_61 -code=sm_61

$(EXE) : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

##$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cpp
##	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu
	$(CC) $(NVCCFLAGS) $(IDIR) -c -o $@ $<

clean:
	rm -f *.o $(EXE)

re: clean
	make


