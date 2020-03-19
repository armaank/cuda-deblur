CC = g++ # compiler
CXXFLAGS = -std=c++11 # flags
PNG_LIB = `libpng-config --ldflags`

make: main.cpp cpuDeblur.o benchmarks.o ops.o pngConnector.o
	$(CC) $(CXXFLAGS) main.cpp cpuDeblur.cpp ../utils/ops.cpp ../benchmarks/metrics.cpp ../utils/pngConnector.cpp $(PNG_LIB) -o cpuDeblurImage

debug: main.cpp cpuDeblur.o benchmarks.o ops.o
	$(CC) $(CXXFLAGS) -g main.cpp cpuDeblur.cpp ../utils/ops.cpp ../benchmarks/metrics.cpp ../utils/pngConnector.cpp $(PNG_LIB)

cpuDeblur.o: cpuDeblur.cpp cpuDeblur.hpp ops.o
	$(CC) $(CXXFLAGS) -g -c cpuDeblur.cpp ../utils/ops.cpp 

ops.o: ../utils/ops.cpp ../utils/ops.hpp
	$(CC) $(CXXFLAGS) -c ../utils/ops.cpp

benchmarks.o: ../benchmarks/metrics.cpp ../benchmarks/metrics.hpp
	$(CC) $(CXXFLAGS) -c ../benchmarks/metrics.cpp

pngConnector.o: ../utils/pngConnector.cpp ../utils/pngConnector.hpp
	$(CC) $(CXXFLAGS) -c ../utils/pngConnector.cpp 

clean:
	rm *.o cpuDeblurImage 
