CC = g++
CXXFLAGS = -std=c++11
PNG_LIB = `libpng-config --ldflags`

make: main.cpp cpuDeblur.o benchmarks.o ops.o pngConnector.o
	$(CC) $(CXXFLAGS) main.cpp cpuDeblur.o ops.o metrics.o pngConnector.o $(PNG_LIB) -o cpu_deblur.out

debug: main.cpp cpuDeblur.o benchmarks.o ops.o
	$(CC) $(CXXFLAGS) -g main.cpp cpuDeblur.o ops.o metrics.o pngConnector.o $(PNG_LIB) -o cpu_deblur_debug.out

cpuDeblur.o: cpuDeblur.cpp cpuDeblur.hpp ops.o
	$(CC) $(CXXFLAGS) -c cpuDeblur.cpp ../utils/ops.cpp 

ops.o: ../utils/ops.cpp ../utils/ops.hpp
	$(CC) $(CXXFLAGS) -c ../utils/ops.cpp

benchmarks.o: ../benchmarks/metrics.cpp ../benchmarks/metrics.hpp
	$(CC) $(CXXFLAGS) -c ../benchmarks/metrics.cpp

pngConnector.o: ../utils/pngConnector.cpp ../utils/pngConnector.hpp
	$(CC) $(CXXFLAGS) -c ../utils/pngConnector.cpp 

clean:
	rm *.o cpu_deblur.out cpu_deblur_debug.out
