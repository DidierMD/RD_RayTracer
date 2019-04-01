#include <iostream>
#include "MyML.h"
#include "RayTracer.h"
#include "FreeImage.h"

int main(int argc, char* argv[]) 
{
	if (argc < 2) {
	std::cerr << "Usage: argc for input scenefile\n"; 
	exit(-1); 
	}
	RayTracer <double> r_t;
	r_t.readFile(argv[1]); 
	r_t.render();
	r_t.saveImage();

	return 0;
}
