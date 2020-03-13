CC = g++ # compiler
CXXFLAGS = -std=c++11 # flags
PNG_LIB = `libpng-config --ldflags`

make: main.cpp cpuLucyRichardson.o benchmarks.o
	$(CC) $(CXXFLAGS) main.cpp png_lib/lodepng.cpp cpuLucyRichardson.cpp ../benchmarks/metrics.cpp $(PNG_LIB) -o deblur.out

debug: main.cpp cpuLucyRichardson.o benchmarks.o
	$(CC) $(CXXFLAGS) -g main.cpp png_lib/lodepng.cpp cpuLucyRichardson.cpp ../benchmarks/metrics.cpp $(PNG_LIB)

cpuLucyRichardson.o: cpuLucyRichardson.cpp cpuLucyRichardson.hpp
	$(CC) $(CXXFLAGS) -c cpuLucyRichardson.cpp

benchmarks.o: ../benchmarks/metrics.cpp ../benchmarks/metrics.hpp
	$(CC) $(CXXFLAGS) -c ../benchmarks/metrics.cpp

clean:
	rm *.o *.out