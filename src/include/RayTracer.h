// Ray Tracer by Didier Muñoz

#ifndef RAY_TRACER
#define RAY_TRACER

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <thread>
#include <chrono>
#include <FreeImage.h>
#include "MyML.h"
#include "DobleFor.h"
#include "SceneDescription.h"
#include "AccelerationStructures.h"

using myml::Vec3;
using myml::Vec4;
using myml::Mat3;
using myml::Mat4;
using myml::zero;
using myml::one;

///////////CameraImage

template <typename T>
class Image{
	private:
		Vec3<T>* Data;
		int Width;
		int Height;
		
	public:
		Image(void);
		Image(int w_, int h_);
		~Image(void);
		static BYTE toByte(const T& color);
		void saveToFile(const std::string& file_name) const;

		int width (void) const {return Width;};
		int height(void) const {return Height;};
		Vec3<T>* operator [](int i){return Data + Width * i;};
		const Vec3<T>* operator [](int i) const {return Data + Width * i;};
		Image<T>& operator = (const Image<T>& der);
};


/////////////Scene

// Scene must hold all the description information

template <typename T, typename ACCL>
class Scene{
	private:
		std::vector <DirectionalLight<T> > DirectionalLights;
		std::vector <PointLight<T> > PointLights;
		std::vector<Light<T>*> LightsPtrs;

		Geometry<T> Primitives;
		ACCL AccelStructure;

		inline bool intersect(const Ray<T>&, Intersection<T>& hit_info) const;
		inline Vec3<T> computeLight(const Vec3<T>& eye_direction, const Vec3<T>& light_direction,
									const Object<T>* hit_obj, const Vec3<T>& normal_, const Light<T>* light)const;
	
	public:
		inline Scene(void);
		inline void add(const DirectionalLight<T>& light_);
		inline void add(const PointLight<T>& light_);
		inline void add(const Triangle<T>& tri_);
		inline void add(const Sphere<T>& sph_);

		inline void createStructures(void);
		inline Vec3<T> calculateColor(const Ray<T>&, int refl_depth);
};

////////////RayTracer

template<typename T>
class RayTracer{
	private:
		Camera<T> ActualCamera;
		Image<T> ActualImage;
		Scene<T, NaiveAccelStruct<T> > ActualScene;

		int MaxReflectionsDepth;

		
		static bool readvals (std::stringstream &s, const int numvals, T * values) ;

	public:
		RayTracer(void);
		void renderPixel_thread(DobleFor& dfor_);
		void render(void);
		void readFile(const std::string& file_name);
		void saveImage(const std::string& file_name) const;

		void setMaxReflectionsDepth(const int& dep);
		int maxReflectionsDepth(void)const{return MaxReflectionsDepth;};

};

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////IMPLEMENTATION/////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////


///////////Image

template <typename T>
Image<T>::Image(void){
	Data = NULL;
	Width = zero<T>();
	Height = zero<T>();
}

template <typename T>
Image<T>::Image(int w_, int h_){
	Data = new Vec3<T>[w_ * h_];
	Width = w_;
	Height = h_;
}

template <typename T>
Image<T>::~Image(void){
	delete[] Data;
}

template <typename T>
BYTE Image<T>::toByte(const T& color){
	T aux = color;
	if (aux > myml::one<T>()) aux = myml::one<T>();
	if (aux < myml::zero<T>()) aux = myml::zero<T>();
	return static_cast<BYTE>(aux * static_cast<T>(255.));
}

template <typename T>
void Image<T>::saveToFile(const std::string& file_name) const{
	BYTE pixels[3 * width() * height()];   
	for(int i=0; i<height(); i++)
		for(int j=0; j<width(); j++){
			pixels[3*(i*width() + j)]  =  toByte((*this)[i][j][2]);
			pixels[3*(i*width() + j) + 1] = toByte((*this)[i][j][1]);
			pixels[3*(i*width() + j) + 2] = toByte((*this)[i][j][0]);
		}

	FIBITMAP *img = FreeImage_ConvertFromRawBits(pixels, width(), height(), width() * 3, 24, 
													  0xFF0000, 0x00FF00, 0x0000FF, false);

	std::cout << "\nSaving image: " << file_name << std::endl;

	FreeImage_Save(FIF_TIFF, img, file_name.c_str(), 0);
}

template <typename T>
Image<T>& Image<T>::operator = (const Image<T>& der){
	if(width() != der.width() || height() != der.height()){
		if(Data != NULL)
			delete[] Data;
		Data = new Vec3<T>[der.width() * der.height()];
		Width = der.width();
		Height = der.height();
	}
	for(int i=0; i<height(); i++)
		for(int j=0; j<width(); j++)
			(*this)[i][j] = der[i][j];
	return *this;
}


///////////Scene

template <typename T, typename ACCL> inline
bool Scene<T, ACCL>::intersect(const Ray<T>& ray_, Intersection<T>& best_hit_info) const{
	return AccelStructure.intersectionSearch(ray_, best_hit_info);
}

template <typename T, typename ACCL> inline 
Scene<T, ACCL>::Scene(void){
}

template <typename T, typename ACCL> inline
void Scene<T, ACCL>::add(const DirectionalLight<T>& light_){
	DirectionalLights.push_back(light_);
}

template <typename T, typename ACCL> inline
void Scene<T, ACCL>::add(const PointLight<T>& light_){
	PointLights.push_back(light_);
}

template <typename T, typename ACCL> inline 
void Scene<T, ACCL>::add(const Triangle<T>& tri_){
	Primitives.add(tri_);
}

template <typename T, typename ACCL> inline 
void Scene<T, ACCL>::add(const Sphere<T>& sph_){
	Primitives.add(sph_);
}

template <typename T, typename ACCL> inline
void Scene<T, ACCL>::createStructures(void){
	LightsPtrs.clear();
	int tam = DirectionalLights.size();
	for(int i = 0; i<tam; i++)
		LightsPtrs.push_back(&DirectionalLights[i]);

	tam = PointLights.size();
	for(int i = 0; i<tam; i++)
		LightsPtrs.push_back(&PointLights[i]);
	
	AccelStructure.constructStructure(Primitives);
}

template <typename T, typename ACCL> inline
Vec3<T> Scene<T, ACCL>::computeLight(const Vec3<T>& eye_direction, const Vec3<T>& light_direction, 
                         		 const Object<T>* hit_obj, const Vec3<T>& normal_, const Light<T>* light_)const{
	Vec3<T> half_vec = myml::normalize(eye_direction + light_direction);
	
	Vec3<T> lambert = std::max(myml::dotProd(normal_, light_direction), zero<T>()) * hit_obj->diffuse();
	Vec3<T> phong = static_cast<T>(pow(std::max(myml::dotProd(normal_, half_vec), zero<T>()), hit_obj->shininess())) *
						 hit_obj->specular();
	return (lambert + phong) * light_->color();
}

template <typename T, typename ACCL> inline
Vec3<T> Scene<T, ACCL>::calculateColor(const Ray<T>& ray_, int refl_depth_){
	Vec3<T> col;
	if(refl_depth_ < 0) return col;

	Intersection<T> hit_info;
	if(false == intersect(ray_, hit_info))
		return col;

	Vec3<T> hitpoint = ray_.alongRay(hit_info.tParam());
	Intersection<T> hit_aux;
	int tam = LightsPtrs.size();
	for(int i=0; i<tam; i++){ // Calculate each light contribution to the color
		// For each light, get ray to light
		Ray<T> raytolight;
		T light_t = LightsPtrs[i] -> getRay(hitpoint, raytolight);
		// Search for an intersection
		hit_aux.setTParam(myml::infinity<T>());
		bool hits = intersect(raytolight, hit_aux);
		// If the ray does not hit or if it hits beyond the light: calculate color due to the light
		if(false == hits || hit_aux.tParam() > light_t){
			col = col + LightsPtrs[i] -> attenuation(hitpoint) * 
																	computeLight(-ray_.direction(), raytolight.direction(),
																	hit_info.hitObject(), hit_info.normalVec(), LightsPtrs[i]);
		}
	}
	// Calculate the reflection contribution to the color. First, I get the ray to work with
	Ray<T> refl_ray = reflectionRay(hitpoint, hit_info.normalVec(), ray_.direction());
	// Then I recursively calculate the color 
	Vec3<T> reflectioncol = hit_info.hitObject()->specular() * calculateColor(refl_ray, refl_depth_-1);
	// Finally, the color returned is the sum of the calculated colors.
	return col + hit_info.hitObject()->emission() + hit_info.hitObject()->ambient() + reflectioncol;
}

///////////RayTracer

template <typename T>
bool RayTracer<T>::readvals(std::stringstream &s, const int numvals, T* values) 
{
  for (int i = 0; i < numvals; i++) {
    s >> values[i]; 
    if (s.fail()) {
      std::cerr << "Failed reading value " << i << " will skip\n"; 
      return false;
    }
  }
  return true; 
}

template <typename T>
RayTracer<T>::RayTracer(void) : MaxReflectionsDepth(5){
}

template <typename T>
void RayTracer<T>::setMaxReflectionsDepth(const int& dep){
	MaxReflectionsDepth = dep;
}

template <typename T>
void RayTracer<T>::renderPixel_thread(DobleFor& dfor_){
	int i,j;
	while(dfor_.getIndexes(i,j)){
		Ray<T> main_ray(ActualCamera, i, j);
		ActualImage[i][j] = ActualScene.calculateColor(main_ray, maxReflectionsDepth());
	}
}

template <typename T>
void RayTracer<T>::render(void){
	ActualScene.createStructures(); // Create acceleration structures

	std::vector<std::thread> hilos;
   DobleFor dfor(0,0, ActualImage.height(), ActualImage.width());

   int concurrency = std::thread::hardware_concurrency();
   std::cout << concurrency << " hilos" << std::endl;

   for (int k=1; k < concurrency; ++k) // Desde k=1 para que sea concurrency-1 hilos extra
      hilos.push_back(std::thread(&RayTracer::renderPixel_thread, this, std::ref(dfor)));
   // Se usa este hilo también para calcular
	renderPixel_thread(dfor);

   for(auto& th : hilos)
      th.join();
}

template <typename T>
void RayTracer<T>::readFile(const std::string& file_name){
	std::string str, cmd; 
	std::ifstream in;
	in.open(file_name); 
	if (in.is_open()) {
    	std::stack <Mat4<T> > transfstack; 
		std::stack <Mat4<T> > invtransfstack;
    	transfstack.push(Mat4<T>(1.0));  // identity
    	invtransfstack.push(Mat4<T>(1.0)); 
		std::vector <Vec3<T> > vertexstack; // vertexes
		Vec3<T> attenuation(one<T>(), zero<T>(), zero<T>());
		Vec3<T> ambient(.2,.2,.2);
		Vec3<T> diffuse;
		Vec3<T> specular;
		Vec3<T> emission;
		T shininess;
    	getline (in, str); 
    	while (in) {
      	if ((str.find_first_not_of(" \t\r\n") != std::string::npos) && (str[0] != '#')) {
        	// Ruled out comment and blank lines 
        	std::stringstream s(str);
        	s >> cmd; 
        	T values[10]; // Position and color for light, colors for others
        	// Up to 10 params for cameras.  
        	bool validinput; // Validity of input 

        	// Process the light, add it to database.
        	// Lighting Command
        	if (cmd == "directional") {
            validinput = readvals(s, 6, values); // direction/color for lts.
            if (validinput) {
					Vec4<T> aux(values[0],values[1],values[2], zero<T>());
					aux = transfstack.top() * aux;
					ActualScene.add(DirectionalLight<T>( myml::toVec3(aux), 
														Vec3<T>(values[3],values[4],values[5]), attenuation[0]));
          	}
			}
			else if (cmd == "point") {
            validinput = readvals(s, 6, values); // position/color for lts.
            if (validinput) {
					Vec4<T> aux(values[0],values[1],values[2], one<T>());
					aux = transfstack.top() * aux;
					ActualScene.add(PointLight<T>( myml::toVec3(aux), 
														Vec3<T>(values[3],values[4],values[5]), attenuation));
          	}
			} 
			else if (cmd == "attenuation") {
            validinput = readvals(s, 3, values);
            if (validinput) {
					attenuation = Vec3<T>(values[0],values[1],values[2]);
          	}
			}

			// Material Commands 

			else if (cmd == "ambient") {
          	validinput = readvals(s, 3, values); // colors 
          	if (validinput) {
					ambient = Vec3<T>(values[0], values[1], values[2]);
            }
         }
			else if (cmd == "diffuse") {
          	validinput = readvals(s, 3, values); // colors 
          	if (validinput) {
					diffuse = Vec3<T>(values[0], values[1], values[2]);
            }
        	} 
			else if (cmd == "specular") {
          	validinput = readvals(s, 3, values); // colors 
          	if (validinput) {
					specular = Vec3<T>(values[0], values[1], values[2]);
            }
        	} 
			else if (cmd == "emission") {
          	validinput = readvals(s, 3, values); // colors 
          	if (validinput) {
					emission = Vec3<T>(values[0], values[1], values[2]);
            }
        	} 
			else if (cmd == "shininess") {
          	validinput = readvals(s, 1, values); 
          	if (validinput) {
            	shininess = values[0]; 
          	}
        	} 

			//Camera properties

			else if (cmd == "size") {
          	validinput = readvals(s,2,values); 
          	if (validinput) { 
					ActualImage = Image<T>((int)values[0], (int)values[1]);
          	} 
        	} 
			else if (cmd == "maxdepth") {
          	validinput = readvals(s,1,values); 
          	if (validinput) { 
					setMaxReflectionsDepth(static_cast<int>(values[0]));
          	} 
        	} 
			else if (cmd == "camera") {
          	validinput = readvals(s,10,values); // 10 values eye cen up fov
          	if (validinput) {
					// camera lookfromx lookfromy lookfromz lookatx lookaty lookatz upx upy upz fovy
					ActualCamera = Camera<T>(Vec3<T>(values[0], values[1], values[2]), //lookfrom
														Vec3<T>(values[3], values[4], values[5]), //lookat
											 			Vec3<T>(values[6], values[7], values[8]), values[9], //up and fovy
											 			ActualImage.width(), ActualImage.height()); //image size
          	}
        	}
			else if (cmd == "vertex"){
				validinput = readvals(s, 3, values);
				if (validinput){
					vertexstack.push_back(Vec3<T>(values[0], values[1], values[2]));
				}
			}
			else if (cmd == "tri"){
				validinput = readvals(s, 3, values);
				if (validinput){
					Vec4<T> vert1 = myml::homogeneous(vertexstack[(int)values[0]]);
					Vec4<T> vert2 = myml::homogeneous(vertexstack[(int)values[1]]);
					Vec4<T> vert3 = myml::homogeneous(vertexstack[(int)values[2]]);
					vert1 = transfstack.top() * vert1;
					vert2 = transfstack.top() * vert2;
					vert3 = transfstack.top() * vert3;
					ActualScene.add(Triangle<T>(ambient, diffuse, specular, emission, shininess,
									myml::toVec3(vert1), myml::toVec3(vert2), myml::toVec3(vert3)));
				}
			}
			else if (cmd == "maxverts"){
				//std::cout << "maxverts read" << std::endl;
			}
        else if (cmd == "sphere") {
            validinput = readvals(s, 4, values); 
            if (validinput) {
					ActualScene.add(Sphere<T>(ambient, diffuse, specular, emission, shininess,
							Vec3<T>(values[0], values[1], values[2]), values[3], transfstack.top(), invtransfstack.top()));
            }
        }
        else if (cmd == "translate") {
          validinput = readvals(s,3,values); 
          if (validinput) {
				Mat4<T> trans_mat = myml::translateMat(values[0], values[1], values[2]);
				Mat4<T> inv_trans_mat = myml::translateMat(-values[0], -values[1], -values[2]);
				transfstack.top() = transfstack.top() * trans_mat;
				invtransfstack.top() = inv_trans_mat * invtransfstack.top();
          }
        }
        else if (cmd == "scale") {
          validinput = readvals(s,3,values); 
          if (validinput) {
				Mat4<T> scale_mat = myml::scaleMat(values[0], values[1], values[2]);
				Mat4<T> inv_scale_mat = myml::scaleMat(one<T>()/values[0], one<T>()/values[1], one<T>()/values[2]);
				transfstack.top() = transfstack.top() * scale_mat;
				invtransfstack.top() = inv_scale_mat * invtransfstack.top();
          }
        }
        else if (cmd == "rotate") {
          validinput = readvals(s,4,values); 
          if (validinput) {
				Vec4<T> aux(values[0], values[1], values[2], zero<T>());
				Mat4<T> rot_mat = myml::rotateMat(values[3], aux);
				Mat4<T> inv_rot_mat = myml::rotateMat(-values[3], aux);
				transfstack.top() = transfstack.top() * rot_mat;
				invtransfstack.top() = inv_rot_mat * invtransfstack.top();
          }
        }
        else if (cmd == "pushTransform") {
          transfstack.push(transfstack.top()); 
          invtransfstack.push(invtransfstack.top()); 
        } else if (cmd == "popTransform") {
          	if (transfstack.size() <= 1) {
            	std::cerr << "Stack has no elements.  Cannot Pop\n"; 
          	} 
				else {
            	transfstack.pop(); 
            	invtransfstack.pop(); 
          	}
        }

        else {
          std::cerr << "Unknown Command: " << cmd << " Skipping \n"; 
        }
      }
      getline (in, str); 
    }
  } else {
    std::cerr << "Unable to Open Input Data File " << file_name << "\n"; 
    throw 2; 
  }
}	

template <typename T>
void RayTracer<T>::saveImage(const std::string& filename) const{
	ActualImage.saveToFile(filename);
}


#endif
