CC = g++ # compiler
CXXFLAGS = -std=c++11 # flags
PNG_LIB = `libpng-config --ldflags`

make: main.cpp cpuLucyRichardson.o benchmarks.o
	$(CC) $(CXXFLAGS) main.cpp cpuLucyRichardson.o metrics.o $(PNG_LIB) -o deblur.out

debug: main.cpp cpuLucyRichardson.o benchmarks.o
	$(CC) $(CXXFLAGS) -g main.cpp cpuLucyRichardson.o metrics.o $(PNG_LIB)

cpuLucyRichardson.o: cpuLucyRichardson.cpp cpuLucyRichardson.hpp
	$(CC) $(CXXFLAGS) -c cpuLucyRichardson.cpp

benchmarks.o: ../benchmarks/metrics.cpp ../benchmarks/metrics.hpp
	$(CC) $(CXXFLAGS) -c ../benchmarks/metrics.cpp

clean:
	rm *.o *.out