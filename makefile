Main: Main.cpp MyML.h RayTracer.h MyML.cpp DobleFor.h RayTracer.cpp
	c++ -std=c++11 -stdlib=libc++ -Wall -O2 -o Main Main.cpp -L. -lfreeimage

#RayTracer.o: RayTracer.cpp RayTracer.h MyML.o
#	g++ -Wall -c RayTracer.cpp

#MyML.o: MyML.h MyML.cpp
#	g++ -Wall -c MyML.cpp
