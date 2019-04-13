// Acceleration Structures file by Didier Muñoz

// This header file is meant to contain the accelaration structures and algorithms (KD-Tree, BVH, etc) for the Ray Tracer.

#ifndef ACCELERATION_STRUCTURES
#define ACCELERATION_STRUCTURES

#include <vector>
#include "SceneDescription.h"

////////////////////KD-Tree grid

class KDTreeGrid{
public:
	KDTreeGrid(void){};
	inline void constructStructure(const Geometry<T>& primitives);

private:
	std::vector<const Object<T>*> ObjectsPtrs;
};


////////////////////Naive

template <typename T>
class NaiveAccelStruct{
public:
	NaiveAccelStruct(void){};
	inline void constructStructure(const Geometry<T>& primitives);
	inline bool intersectionSearch(const Ray<T>& ray_, Intersection<T>& hit_info) const;

private:
	std::vector<const Object<T>*> ObjectsPtrs;
};

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////IMPLEMENTATION/////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

//////////////////////KD-Tree Grid

template <typename T> inline 
void KDTreeGrid<T> :: constructStructure(const Geometry<T>& primitives){
	
}


//////////////////////Naive

template <typename T> inline 
void NaiveAccelStruct<T> :: constructStructure(const Geometry<T>& primitives){
	ObjectsPtrs.clear();
	unsigned tam = primitives.Triangles.size();
	for(int i = 0; i<tam; i++)
		ObjectsPtrs.push_back(&(primitives.Triangles[i]));

	tam = primitives.Spheres.size();
	for(int i = 0; i<tam; i++)
		ObjectsPtrs.push_back(&primitives.Spheres[i]);
}

template<typename T> inline 
bool NaiveAccelStruct<T> :: intersectionSearch(const Ray<T>& ray_, Intersection<T>& best_hit_info) const{
	Intersection<T> hit_info;
	bool hits = false;

	best_hit_info.setTParam(myml::infinity<T>());
	int tam = ObjectsPtrs.size();
	for (int i=0; i<tam; i++){
		if(ObjectsPtrs[i] -> intersect(ray_, hit_info)){
			if(hit_info.tParam() < best_hit_info.tParam()){
				best_hit_info = hit_info;
				hits = true;
			}
		}
	}
	return hits;
}

#endif
