CC = g++ # compiler
CXXFLAGS = -std=c++11 # flags

make: cpuLucyRichardson.o
	$(CC) $(CXXFLAGS)  main.cpp png_lib/lodepng.cpp cpuLucyRichardson.cpp

debug: cpuLucyRichardson.o
	$(CC) $(CXXFLAGS)  -g main.cpp png_lib/lodepng.cpp cpuLucyRichardson.cpp

cpuLucyRichardson.o: cpuLucyRichardson.cpp cpuLucyRichardson.h
	$(CC) $(CXXFLAGS)  -c cpuLucyRichardson.cpp

clean:
	rm cpuLucyrichardson.o a.out