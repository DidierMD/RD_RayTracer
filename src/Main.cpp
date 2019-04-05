#include <iostream>
#include <FreeImage.h>
#include "MyML.h"
#include "RayTracer.h"

int main(int argc, char* argv[]) 
{
	if (argc < 3) {
		std::cerr << "Usage: RD_RT <scene_description> <output_file>\n"; 
		return -1; 
	}
	RayTracer <double> r_t;
	r_t.readFile(argv[1]); 
	r_t.render();
	r_t.saveImage(argv[2]);

	return 0;
}
