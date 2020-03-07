make: cpuLucyRichardson.o
	g++ main.cpp png_lib/lodepng.cpp cpuLucyRichardson.cpp

debug: cpuLucyRichardson.o
	g++ -g main.cpp png_lib/lodepng.cpp cpuLucyRichardson.cpp

cpuLucyRichardson.o: cpuLucyRichardson.cpp cpuLucyRichardson.h
	g++ -c cpuLucyRichardson.cpp

clean:
	rm cpuLucyrichardson.o a.out