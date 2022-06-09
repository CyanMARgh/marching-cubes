FLAGS=-lsfml-window -lsfml-system -lsfml-graphics -lpthread -std=c++17 -lm
DEBUG ?= 1
ifeq ($(DEBUG), 1)
	FLAGS_EXTRA=-DDEBUG
else
	FLAGS_EXTRA=-O3 -Wextra -Wall -DNDEBUG
endif

all: build/app

build/app: build/ temp/main.o temp/SimplexNoise.o
	g++ -o build/app temp/main.o temp/SimplexNoise.o $(FLAGS)

temp/main.o: temp/ main.cpp SimplexNoise.h
	g++ -c main.cpp -o temp/main.o $(FLAGS)

temp/SimplexNoise.o : temp/ SimplexNoise.cpp SimplexNoise.h
	g++ -c SimplexNoise.cpp -o temp/SimplexNoise.o $(FLAGS)

temp/:
	mkdir temp
build/:
	mkdir build