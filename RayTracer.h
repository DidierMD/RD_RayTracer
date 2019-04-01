// Ray Tracer by Didier Muñoz

#ifndef RAY_TRACER
#define RAY_TRACER

#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <thread>
#include <chrono>
#include "MyML.h"
#include "FreeImage.h"
#include "DobleFor.h"

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

///////////Camera

template<typename T>
class Camera{
	private:
		Vec3<T> LookFrom;
		Vec3<T> LookAt;
		Vec3<T> Up;
		T Fovy2;
		T Fovx2;
		T Width2;
		T Height2;
		Vec3<T> U;
		Vec3<T> V;
		Vec3<T> W;
		
	public:
		Camera(void);
		Camera(const Vec3<T>& look_from, const Vec3<T>& look_at, const Vec3<T>& up_, const T& fovy_, const T& w_, const T& h_);
		const Vec3<T>& lookFrom(void) const{return LookFrom;};
		const Vec3<T>& lookAt(void) const{return LookAt;};
		const Vec3<T>& up(void) const{return Up;};
		const T& fovy2(void) const{return Fovy2;};
		const T& fovx2(void) const{return Fovx2;};
		const T& width2(void) const{return Width2;};
		const T& height2(void) const{return Height2;};
		const Vec3<T>& u(void) const{return U;};
		const Vec3<T>& v(void) const{return V;};
		const Vec3<T>& w(void) const{return W;};
};

////////////Ray

template <typename T>
class Ray{
	private:
		Vec3<T> Origin;
		Vec3<T> Direction;
		
	public:
		Ray(void){};
		Ray(const Camera<T>& cam, int i, int j);
		Ray(const Vec3<T>& origin_, const Vec3<T>& direction_);
		const Vec3<T>& origin(void) const{return Origin;};
		const Vec3<T>& direction(void) const{return Direction;};
		Vec3<T> alongRay(const T& t_)const;
};

template <typename T>
Ray<T> reflectionRay(const Vec3<T>& hitpoint, const Vec3<T>& normal, const Vec3<T>& direction);

/////////////forward declaration of Object

template <typename T>
class Object;

/////////////Intersection

template <typename T>
class Intersection{
	private:
		T TParam;
		Vec3<T> NormalVec;
		const Object<T>* HitObject;

	public:
		void setTParam(const T& t_){TParam = t_;};
		void setNormalVec(const Vec3<T>& normal_){NormalVec = normal_;};
		void setHitObject(const Object<T>* obj_ptr){HitObject = obj_ptr;};
		T tParam(void) const {return TParam;};
		Vec3<T> normalVec(void) const {return NormalVec;};
		const Object<T>* hitObject(void) const {return HitObject;};
};

/////////////Objects

template <typename T>
class Object{
	private:
		Vec3<T> Ambient;
		Vec3<T> Diffuse;
		Vec3<T> Specular;
		Vec3<T> Emission;
		T Shininess;
		
	public:
		Object(const Vec3<T>& amb_, const Vec3<T>& dif_, const Vec3<T>& spe_, const Vec3<T>& emi_, const T& shi_);
		virtual ~Object(void){};
		const Vec3<T>& ambient(void) const {return Ambient;};
		const Vec3<T>& diffuse(void) const {return Diffuse;};
		const Vec3<T>& specular(void) const {return Specular;};
		const Vec3<T>& emission(void) const {return Emission;};
		const T& shininess(void) const {return Shininess;};
		virtual bool intersect(const Ray<T>& ray_, Intersection<T>& hit_info) const = 0; 
};

template <typename T>
class Triangle : public Object<T>{
	private:
		Vec3<T> Vertices[3];

	public:
		Triangle(const Vec3<T>& amb_, const Vec3<T>& dif_, const Vec3<T>& spe_, const Vec3<T>& emi_, const T& shi_, 
					const Vec3<T>& vert0, const Vec3<T>& vert1, const Vec3<T>& vert2);
		bool intersect(const Ray<T>& ray_, Intersection<T>& hit_info) const;
};

template <typename T>
class Sphere : public Object<T>{
	private:
		Vec3<T> Center;	
		T Radius;
		Mat4<T> Transform;
		Mat4<T> InvTransform;

	public:
		Sphere(const Vec3<T>& amb_, const Vec3<T>& dif_, const Vec3<T>& spe_, const Vec3<T>& emi_, const T& shi_, 
					const Vec3<T>& center_, const T& radius_, const Mat4<T>& trans_, const Mat4<T>& inv_trans);
		bool intersect(const Ray<T>& ray_, Intersection<T>& hit_info) const;
		
};

////////////Light

template <typename T>
class Light{
	private:
		Vec3<T> Color;
	public:
		Light(void){};
		Light(const Vec3<T>& col){Color=col;};
		virtual ~Light(void){};
		virtual T getRay(const Vec3<T>&, Ray<T>&) const = 0;
		virtual T attenuation(const Vec3<T>&) const = 0;
		Vec3<T> color(void)const{return Color;};
};

template <typename T>
class DirectionalLight : public Light<T>{
	private:
		Vec3<T> Direction;
		T Attenuation;
		
	public:
		DirectionalLight(const Vec3<T>& dir, const Vec3<T>& col, const T& attenuation_);
		T getRay(const Vec3<T>&, Ray<T>&)const;
		T attenuation(const Vec3<T>&) const;
};

template <typename T>
class PointLight : public Light<T>{
	private:
		Vec3<T> Position;
		Vec3<T> Attenuation;

	public:
		PointLight(const Vec3<T>& pos, const Vec3<T>& col, const Vec3<T>& attenuation_);
		T getRay(const Vec3<T>&, Ray<T>&) const;
		T attenuation(const Vec3<T>&) const;
};

/////////////Scene

// Scene must hold all the description information

template <typename T>
class Scene{
	private:
		std::vector <DirectionalLight<T> > DirectionalLights;
		std::vector <PointLight<T> > PointLights;
		std::vector <Triangle<T> > Triangles;
		std::vector <Sphere<T> > Spheres;
		
		std::vector<Light<T>*> LightsPtrs;
		std::vector<Object<T>*> ObjectsPtrs;

		bool intersect(const Ray<T>&, Intersection<T>& hit_info) const;
		Vec3<T> computeLight(const Vec3<T>& eye_direction, const Vec3<T>& light_direction,
									const Object<T>* hit_obj, const Vec3<T>& normal_, const Light<T>* light)const;
	
	public:
		Scene(void);
		void add(const DirectionalLight<T>& light_);
		void add(const PointLight<T>& light_);
		void add(const Triangle<T>& tri_);
		void add(const Sphere<T>& sph_);

		void createStructures(void);
		Vec3<T> calculateColor(const Ray<T>&, int refl_depth);
};

////////////RayTracer

template<typename T>
class RayTracer{
	private:
		Camera<T> ActualCamera;
		Image<T> ActualImage;
		Scene<T> ActualScene;
		std::string Output;

		int MaxReflectionsDepth;

		
		static bool readvals (std::stringstream &s, const int numvals, T * values) ;

	public:
		RayTracer(void);
		void renderPixel_thread(DobleFor& dfor_);
		void render(void);
		void readFile(const std::string& file_name);
		void saveImage(void) const;

		void setMaxReflectionsDepth(const int& dep);
		int maxReflectionsDepth(void)const{return MaxReflectionsDepth;};

};

#include "RayTracer.cpp"

#endif
