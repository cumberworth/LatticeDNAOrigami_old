CC = g++
CPPFLAGS = -I ../../include -g
LDFLAGS =  -lboost_program_options
BUILD = ../../build/
vpath %.h ../../include
#vpath %.cpp src

TARGET = enumerate

all: $(TARGET)

testing: $(TESTTARGET)

$(TARGET): enumerate.o $(BUILD)origami_system.o $(BUILD)utility.o $(BUILD)nearest_neighbour.o $(BUILD)files.o $(BUILD)json.o $(BUILD)domain.o $(BUILD)simulation.o $(BUILD)parser.o $(BUILD)random_gens.o $(BUILD)movetypes.o $(BUILD)ideal_random_walk.o
	$(CC) -o $@ $^ $(LDFLAGS)

$(BUILD)enumerate.o: enumerate.cpp
	$(CC) -o $@ -c $(CPPFLAGS) $<

.PHONY: clean
clean:
	rm *.o
