// SceneDescription by Didier Muñoz

#ifndef SCENE_DESCRIPTION
#define SCENE_DESCRIPTION

#include <vector>
#include <algorithm>
#include <iostream>
#include "MyML.h"

using myml::Vec3;
using myml::Vec4;
using myml::Mat3;
using myml::Mat4;
using myml::zero;
using myml::one;

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

/////////////Geometry

// This structure is meant to hold the existing geometry for the scene

template <typename T>
struct Geometry{
	std::vector <Triangle<T> > Triangles;
	std::vector <Sphere<T> > Spheres;

	inline void add(const Triangle<T>& tri_);
	inline void add(const Sphere<T>& sph_);
};

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////IMPLEMENTATION/////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////


///////////Camera


template <typename T>
Camera<T>::Camera(void){
//Camera<T>::Camera(void) : LookAt(zero<T>(), zero<T>(), -one<T>()), Up(zero<T>(), one<T>(), zero<T>()),
//									 LookFrom(), static_cast<T>(45.), 500, 500)
}


template <typename T>
Camera<T>::Camera(const Vec3<T>& look_from, const Vec3<T>& look_at, const Vec3<T>& up_, 
						const T& fovy_, const T& w_, const T& h_){
	LookFrom = look_from;
	LookAt = look_at;
	Up = up_;
	Fovy2 = myml::radians(fovy_ / static_cast<T>(2.));
	Width2 = w_/static_cast<T>(2.);
	Height2 = h_/static_cast<T>(2.);
	T ratio = w_ / h_;
	Fovx2 = static_cast<T>( atan( ratio*static_cast<T>(tan(Fovy2)) ) );
	W = myml::normalize(LookFrom - LookAt);
	U = myml::normalize(myml::crossProd(Up,W));
	V = myml::crossProd(W, U);
}

///////////Ray

template <typename T>
Ray<T>::Ray(const Camera<T>& cam, int i, int j){
	Origin = cam.lookFrom();
	T alpha = static_cast<T>(tan(cam.fovx2())) * (static_cast<T>(0.5)+static_cast<T>(j)-cam.width2()) / cam.width2();
	T beta = static_cast<T>(tan(cam.fovy2())) * (static_cast<T>(0.5)+static_cast<T>(i)-cam.height2()) / cam.height2();
	Direction = myml::normalize(alpha*cam.u() + beta*cam.v() - cam.w());
}

template <typename T>
Ray<T>::Ray(const Vec3<T>& origin_, const Vec3<T>& direction_){
	Origin = origin_;
	Direction = myml::normalize(direction_);
}

template <typename T>
Vec3<T> Ray<T>::alongRay(const T& t_)const{
	return Origin + (t_ * Direction);
}

template <typename T>
Ray<T> reflectionRay(const Vec3<T>& hitpoint_, const Vec3<T>& normal_, const Vec3<T>& direction_){
	Vec3<T> auxdir = direction_ - static_cast<T>(2.) * myml::dotProd(normal_, direction_) * normal_;
	Ray<T> auxray = Ray<T>(hitpoint_, auxdir);
	return Ray<T>(auxray.alongRay(static_cast<T>(0.001)), auxdir);
}

///////////Objects

template <typename T>
Object<T>::Object(const Vec3<T>& amb_, const Vec3<T>& dif_, const Vec3<T>& spe_, const Vec3<T>& emi_, const T& shi_){
	Ambient = amb_;
	Diffuse = dif_;
	Specular = spe_;
	Emission = emi_;
	Shininess = shi_;
}

template <typename T>
Triangle<T>::Triangle(const Vec3<T>& amb, const Vec3<T>& dif, const Vec3<T>& spe, const Vec3<T>& emi, const T& shi, 
							const Vec3<T>& vert0, const Vec3<T>& vert1, const Vec3<T>& vert2)
							: Object<T>(amb, dif, spe, emi, shi){
	Vertices[0] = vert0;
	Vertices[1] = vert1;
	Vertices[2] = vert2;
}

template <typename T>
bool Triangle<T>::intersect(const Ray<T>& ray_, Intersection<T>& hit_info) const{
	Vec3<T> ba = Vertices[1] - Vertices[0];
	Vec3<T> ca = Vertices[2] - Vertices[0];
	Vec3<T> n = myml::normalize(myml::crossProd(ba, ca));

	T t_val = myml::dotProd(ray_.direction(), n);
	if(t_val == myml::zero<T>()){
		//std::cerr << "Ray in triangle plane" << std::endl;
		return false;
	}
	t_val = myml::dotProd(n, Vertices[0] - ray_.origin()) / t_val;
	if(t_val < myml::zero<T>()){
		return false;
	}
	Vec3<T> pa = ray_.origin() + (t_val * ray_.direction()) - Vertices[0];
	
	int i1=0;
	int i2=1;
	T gamma = ba[i2]*ca[i1] - ba[i1]*ca[i2];
	if( gamma == myml::zero<T>() ){
		i2 = 2;
		gamma = ba[i2]*ca[i1] - ba[i1]*ca[i2];
		if( gamma == myml::zero<T>() ){
			i1 = 1;
			gamma = ba[i2]*ca[i1] - ba[i1]*ca[i2];
			if( gamma == myml::zero<T>() ){
				std::cerr << "not a triangle, will skip" << std::endl;
				return false;
			}
		}
	}
	gamma = (ba[i2]*pa[i1] - ba[i1]*pa[i2]) / gamma;
	
	int j = 0;
	if(ba[j] == myml::zero<T>()){
		j=1;
		if(ba[j] == myml::zero<T>()){
			j=2;
			if(ba[j] == myml::zero<T>()){
				std::cerr << "not a triangle, will skip" << std::endl;
				return false;
			}
		}
	}
	T beta = (pa[j] - gamma*ca[j]) / ba[j];

	if(beta < myml::zero<T>() || gamma < myml::zero<T>() || (beta+gamma) > myml::one<T>())
		return false;

	hit_info.setTParam(t_val);
	hit_info.setNormalVec(n);
	hit_info.setHitObject(this);
	return true;
}

template <typename T>
Sphere<T>::Sphere(const Vec3<T>& amb, const Vec3<T>& dif, const Vec3<T>& spe, const Vec3<T>& emi, const T& shi, 
							const Vec3<T>& center_, const T& radius_, const Mat4<T>& trans_, const Mat4<T>& inv_trans)
							: Object<T>(amb, dif, spe, emi, shi){
	Center = center_;
	Radius = radius_;
	Transform = trans_;
	InvTransform = inv_trans;
}

template <typename T>
bool Sphere<T>::intersect(const Ray<T>& ori_ray, Intersection<T>& hit_info) const{
	Ray<T> ray_(toVec3(InvTransform * myml::homogeneous(ori_ray.origin())), 
					toVec3(InvTransform * Vec4<T>(ori_ray.direction())));
	T a_ = myml::dotProd(ray_.direction(), ray_.direction());
	Vec3<T> p0c = ray_.origin() - Center;
	T b_ = static_cast<T>(2.) * myml::dotProd(ray_.direction(), p0c);
	T c_ = myml::dotProd(p0c, p0c) - Radius*Radius;
	//Discriminante
	T discriminant = b_*b_ - static_cast<T>(4.)*a_*c_;
	if(discriminant < zero<T>())
		return false;
	
	discriminant = static_cast<T>(sqrt(discriminant));
	T root1 = (-b_ + discriminant) / (static_cast<T>(2.) * a_);
	T root2 = (-b_ - discriminant) / (static_cast<T>(2.) * a_);
	T t_val;
	
	if(root1 >= zero<T>())
		if(root2 >= zero<T>())
			t_val = std::min(root1, root2);
		else
			t_val = root1;
	else
		if(root2 >= zero<T>())
			t_val = root2;
		else
			return false;
	
	Vec3<T> aux2 = ray_.alongRay(t_val);

	Vec4<T> aux3 = Transform * myml::homogeneous(aux2);
	hit_info.setTParam(myml::norm(toVec3(aux3) - ori_ray.origin()));
	//hit_info.setTParam(t_val);

	Vec3<T> normal_ = myml::normalize(aux2 - Center);
	Mat3<T> invtrans = myml::transpose(toMat3(InvTransform));
	normal_ = myml::normalize(invtrans * normal_);
	hit_info.setNormalVec(normal_);
	hit_info.setHitObject(this);
	return true;
}

///////////Light

template <typename T>
DirectionalLight<T>::DirectionalLight(const Vec3<T>& dir, const Vec3<T>& col, const T& atte_) : Light<T>(col){
	Direction = dir;
	Attenuation = atte_;
}

template <typename T>
T DirectionalLight<T>::getRay(const Vec3<T>& point_, Ray<T>& ray_) const{
	Ray<T> auxray = Ray<T>(point_, -Direction);
	ray_  = Ray<T>(auxray.alongRay(static_cast<T>(0.001)), auxray.direction());
	return myml::infinity<T>();
}

template <typename T>
T DirectionalLight<T>::attenuation(const Vec3<T>& pos)const{
	return one<T>() / Attenuation;
}

template <typename T>
PointLight<T>::PointLight(const Vec3<T>& pos, const Vec3<T>& col, const Vec3<T>& atte_) : Light<T>(col){
	Position = pos;
	Attenuation = atte_;
}

template <typename T>
T PointLight<T>::getRay(const Vec3<T>& point_, Ray<T>& ray_) const{
	Vec3<T> aux = Position - point_;
	Ray<T> auxray = Ray<T>(point_, aux);
	ray_  = Ray<T>(auxray.alongRay(static_cast<T>(0.001)), aux);
	return myml::norm(aux);
}

template <typename T>
T PointLight<T>::attenuation(const Vec3<T>& pos)const{
	T dis = myml::norm(Position - pos);
	return one<T>() / (Attenuation[0] + dis*Attenuation[1] + dis*dis*Attenuation[2]);
}

///////////Geometry

template <typename T> inline 
void Geometry<T>::add(const Triangle<T>& tri_){
	Triangles.push_back(tri_);
}

template <typename T> inline
void Geometry<T>::add(const Sphere<T>& sph_){
	Spheres.push_back(sph_);
}

#endif
