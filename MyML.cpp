
#include "MyML.h"

namespace myml{

//////////////////////Classes

//Vec3

template <typename T1>
Vec3<T1>::Vec3(void){
	Data[0] = Data[1] = Data[2] = zero<T1>();
}

template <typename T1>
Vec3<T1>::Vec3(T1 x, T1 y, T1 z) {
	Data[0]=x;
	Data[1]=y;
	Data[2]=z;
}

template <typename T1>
T1& Vec3<T1>::operator [] (int i){
	return Data[i];
}

template <typename T1>
T1 Vec3<T1>::operator [] (int i) const{
	return Data[i];
}

//Vec4

template <typename T1>
Vec4<T1>::Vec4(void){
	Data[0] = Data[1] = Data[2] = Data[3] = zero<T1>();
}

template <typename T1>
Vec4<T1>::Vec4(T1 x, T1 y, T1 z, T1 w) {
	Data[0]=x;
	Data[1]=y;
	Data[2]=z;
	Data[3]=w;
}

template <typename T1>
Vec4<T1>::Vec4(const Vec3<T1>& der){
	Data[0]=der[0];
	Data[1]=der[1];
	Data[2]=der[2];
	Data[3]=zero<T1>();
}

template <typename T1>
T1& Vec4<T1>::operator [] (int i){
	return Data[i];
}

template <typename T1>
T1 Vec4<T1>::operator [] (int i) const{
	return Data[i];
}

//Mat3

template <typename T1>
Mat3<T1>::Mat3(void){
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			Data[i][j] = zero<T1>();
}

template <typename T1>
Mat3<T1>::Mat3(T1 val){
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			if(i==j)
				Data[i][j] = val;
			else
				Data[i][j] = zero<T1>();
		}
	}
}

template <typename T1>
Mat3<T1>::Mat3(T1 val1, T1 val2, T1 val3){
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			if(i != j)
				Data[i][j] = zero<T1>();
		}
	}
	Data[0][0] = val1;
	Data[1][1] = val2;
	Data[2][2] = val3;
}

template <typename T1>
Mat3<T1>::Mat3(T1 v00, T1 v01, T1 v02, 
					T1 v10, T1 v11, T1 v12, 
					T1 v20, T1 v21, T1 v22){
	Data[0][0] = v00;
	Data[0][1] = v01;
	Data[0][2] = v02;
	Data[1][0] = v10;
	Data[1][1] = v11;
	Data[1][2] = v12;
	Data[2][0] = v20;
	Data[2][1] = v21;
	Data[2][2] = v22;
}

template <typename T1>
T1* Mat3<T1>::operator [] (int i){
	return Data[i];
}

template <typename T1>
const T1* Mat3<T1>::operator [] (int i) const{
	return Data[i];
}

//Mat4

template <typename T1>
Mat4<T1>::Mat4(void){
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			Data[i][j] = zero<T1>();
}

template <typename T1>
Mat4<T1>::Mat4(T1 val){
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			if(i==j)
				Data[i][j] = val;
			else
				Data[i][j] = zero<T1>();
		}
	}
}

template <typename T1>
Mat4<T1>::Mat4(T1 val1, T1 val2, T1 val3, T1 val4){
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			if(i != j)
				Data[i][j] = zero<T1>();
		}
	}
	Data[0][0] = val1;
	Data[1][1] = val2;
	Data[2][2] = val3;
	Data[3][3] = val4;
}

template <typename T1>
Mat4<T1>::Mat4(T1 v00, T1 v01, T1 v02, T1 v03, 
	  T1 v10, T1 v11, T1 v12, T1 v13, 
	  T1 v20, T1 v21, T1 v22, T1 v23,
	  T1 v30, T1 v31, T1 v32, T1 v33){
	Data[0][0] = v00;
	Data[0][1] = v01;
	Data[0][2] = v02;
	Data[0][3] = v03;
	Data[1][0] = v10;
	Data[1][1] = v11;
	Data[1][2] = v12;
	Data[1][3] = v13;
	Data[2][0] = v20;
	Data[2][1] = v21;
	Data[2][2] = v22;
	Data[2][3] = v23;
	Data[3][0] = v30;
	Data[3][1] = v31;
	Data[3][2] = v32;
	Data[3][3] = v33;
}

template <typename T1>
T1* Mat4<T1>::operator [] (int i){
	return Data[i];
}

template <typename T1>
const T1* Mat4<T1>::operator [] (int i) const{
	return Data[i];
}

////////////////////////Operations

template <typename T1>
Vec3<T1> operator * (T1 scalar, const Vec3<T1>& myvec){
	return Vec3<T1>(myvec.x() * scalar, myvec.y()*scalar, myvec.z()*scalar);
}

template <typename T1>
Vec4<T1> operator * (T1 scalar, const Vec4<T1>& myvec){
	return Vec4<T1>(myvec.x()*scalar, myvec.y()*scalar, myvec.z()*scalar, myvec.w()*scalar);
}

template <typename T1>
Vec3<T1> operator * (const Vec3<T1>& izq, const Vec3<T1>& der){
	return Vec3<T1>(izq.x()*der.x(), izq.y()*der.y(), izq.z()*der.z());
}

template <typename T1>
Vec4<T1> operator * (const Vec4<T1>& izq, const Vec4<T1>& der){
	return Vec4<T1>(izq.x()*der.x(), izq.y()*der.y(), izq.z()*der.z(), izq.w()*der.w());
}

template <typename T1>
Vec3<T1> operator / (const Vec3<T1>& myvec, T1 scalar){
	return Vec3<T1>(myvec.x()/scalar, myvec.y()/scalar, myvec.z()/scalar);
}

template <typename T1>
Vec4<T1> operator / (const Vec4<T1>& myvec, T1 scalar){
	return Vec4<T1>(myvec.x()/scalar, myvec.y()/scalar, myvec.z()/scalar, myvec.w()/scalar);
}

template <typename T1>
Vec3<T1> operator + (const Vec3<T1>& izq, const Vec3<T1>& der){
	return Vec3<T1>(izq.x()+der.x(), izq.y()+der.y(), izq.z()+der.z());
}

template <typename T1>
Vec4<T1> operator + (const Vec4<T1>& izq, const Vec4<T1>& der){
	return Vec4<T1>(izq.x()+der.x(), izq.y()+der.y(), izq.z()+der.z(), izq.w()+der.w());
}

template <typename T1>
Vec3<T1> operator - (const Vec3<T1>& izq, const Vec3<T1>& der){
	return Vec3<T1>(izq.x()-der.x(), izq.y()-der.y(), izq.z()-der.z());
}

template <typename T1>
Vec4<T1> operator - (const Vec4<T1>& izq, const Vec4<T1>& der){
	return Vec4<T1>(izq.x()-der.x(), izq.y()-der.y(), izq.z()-der.z(), izq.w()-der.w());
}

template <typename T1>
Vec3<T1> operator - (const Vec3<T1>& der){
	return Vec3<T1>(-der.x(), -der.y(), -der.z());
}

template <typename T1>
Vec4<T1> operator - (const Vec4<T1>& der){
	return Vec4<T1>(-der.x(), -der.y(), -der.z(), -der.w());
}

template <typename T1>
Mat3<T1> operator * (const Mat3<T1>& izq, const Mat3<T1>& der){
	Mat3<T1> ret;

	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++){
			T1 value = zero<T1>();
			for(int k=0; k<3; k++)
				value += izq[i][k] * der[k][j];
			ret[i][j] = value;
		}
	return ret;
}

template <typename T1>
Mat4<T1> operator * (const Mat4<T1>& izq, const Mat4<T1>& der){
	Mat4<T1> ret;

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++){
			T1 value = zero<T1>();
			for(int k=0; k<4; k++)
				value += izq[i][k] * der[k][j];
			ret[i][j] = value;
		}
	return ret;
}

template <typename T1>
Mat3<T1> operator + (const Mat3<T1>& izq, const Mat3<T1>& der){
	Mat3<T1> ret;

	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			ret[i][j] = izq[i][j] + der[i][j];

	return ret;
}

template <typename T1>
Mat4<T1> operator + (const Mat4<T1>& izq, const Mat4<T1>& der){
	Mat4<T1> ret;

	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			ret[i][j] = izq[i][j] + der[i][j];

	return ret;
}

template <typename T1>
Mat3<T1> operator * (T1 scalar, const Mat3<T1>& der){
	Mat3<T1> ret;
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			ret[i][j] = scalar * der[i][j];
	return ret;
}

template <typename T1>
Mat4<T1> operator * (T1 scalar, const Mat4<T1>& der){
	Mat4<T1> ret;
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			ret[i][j] = scalar * der[i][j];
	return ret;
}

template <typename T1>
Vec3<T1> operator * (const Mat3<T1>& izq_mat, const Vec3<T1>& der_vec){
	Vec3<T1> ret;
	for(int i=0; i<3; i++){
		T1 value = zero<T1>();
		for(int k=0; k<3; k++)
			value += izq_mat[i][k] * der_vec[k];
		ret[i] = value;
	}
	return ret;
}

template <typename T1>
Vec4<T1> operator * (const Mat4<T1>& izq_mat, const Vec4<T1>& der_vec){
	Vec4<T1> ret;
	for(int i=0; i<4; i++){
		T1 value = zero<T1>();
		for(int k=0; k<4; k++)
			value += izq_mat[i][k] * der_vec[k];
		ret[i] = value;
	}
	return ret;
}

/////////////////////////Functions

template <typename T1>
T1 dotProd(const Vec3<T1>& izq, const Vec3<T1>& der){
	return izq[0]*der[0] + izq[1]*der[1] + izq[2]*der[2];
}

template <typename T1>
T1 dotProd(const Vec4<T1>& izq, const Vec4<T1>& der){
	return izq[0]*der[0] + izq[1]*der[1] + izq[2]*der[2] + izq[3]*der[3];
}

template <typename T1>
Vec3<T1> crossProd(const Vec3<T1>& a, const Vec3<T1>& b){
	return Vec3<T1>(a[1]*b[2] - b[1]*a[2],
						 a[2]*b[0] - a[0]*b[2],
						 a[0]*b[1] - a[1]*b[0]);
}

template <typename T1>
Mat3<T1> outerProd(const Vec3<T1>& izq, const Vec3<T1>& der){
	Mat3<T1> ret;
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			ret[i][j] = izq[i] * der[j];
	return ret;
}

template <typename T1>
T1 norm(const Vec3<T1>& der){
	return static_cast<T1>(sqrt( der.x()*der.x() + der.y()*der.y() + der.z()*der.z() ));
}

template <typename T1>
T1 norm(const Vec4<T1>& der){
	return static_cast<T1>(sqrt( der.x()*der.x() + der.y()*der.y() + der.z()*der.z() + der.w()*der.w() ));
}

template <typename T1>
Vec3<T1> normalize(const Vec3<T1>& der){
	return der / norm(der);
}

template <typename T1>
Vec4<T1> normalize(const Vec4<T1>& der){
	return der / norm(der);
}

template <typename T1>
Vec4<T1> homogeneous(const Vec3<T1>& der){
	return Vec4<T1>(der.x(), der.y(), der.z(), one<T1>());
}

template <typename T1>
Vec3<T1> toVec3(const Vec4<T1>& der){
		return Vec3<T1>(der.x(), der.y(), der.z());
}

template <typename T1>
T1 radians(T1 degrees){
	T1 pi = static_cast<T1>(3.141592653589793238462);
	return degrees * pi / static_cast<T1>(180.);
}

template <typename T1>
Mat3<T1> toMat3(const Mat4<T1>& der_mat){
	Mat3<T1> ret;
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			ret[i][j] = der_mat[i][j];
	return ret;
}

template <typename T1>
Mat4<T1> transpose(const Mat4<T1>& der_mat){
	return Mat4<T1>(der_mat[0][0], der_mat[1][0], der_mat[2][0], der_mat[3][0],
						 der_mat[0][1], der_mat[1][1], der_mat[2][1], der_mat[3][1], 
						 der_mat[0][2], der_mat[1][2], der_mat[2][2], der_mat[3][2], 
						 der_mat[0][3], der_mat[1][3], der_mat[2][3], der_mat[3][3]);
}

template <typename T1>
Mat3<T1> transpose(const Mat3<T1>& der_mat){
	return Mat3<T1>(der_mat[0][0], der_mat[1][0], der_mat[2][0], 
						 der_mat[0][1], der_mat[1][1], der_mat[2][1],  
						 der_mat[0][2], der_mat[1][2], der_mat[2][2]);
}

template <typename T1>
Mat3<T1> starMat(const Vec3<T1> &der){
	return Mat3<T1>(zero<T1>(), -der.z(), der.y(),
						 der.z(), zero<T1>(), -der.x(),
						 -der.y(), der.x(), zero<T1>());
}

template <typename T1>
Mat4<T1> starMat(const Vec4<T1> &der){
	return Mat4<T1>(zero<T1>(), -der.z(), der.y(), zero<T1>(),
						 der.z(), zero<T1>(), -der.x(), zero<T1>(),
						 -der.y(), der.x(), zero<T1>(), zero<T1>(),
						 zero<T1>(),	zero<T1>(), zero<T1>(), one<T1>());
}

template <typename T1>
Mat3<T1> rotateMat(const T1& degrees, const Vec3<T1>& axis){
	T1 theta = radians(degrees);
	Vec3<T1> ax_normed = normalize(axis);

	return static_cast<T1>(cos(theta)) * Mat3<T1>(one<T1>()) + 
			(one<T1>()-static_cast<T1>(cos(theta))) * outerProd(ax_normed, ax_normed) + 
			static_cast<T1>(sin(theta)) * starMat(ax_normed);
}

template <typename T1>
Mat4<T1> rotateMat(const T1& degrees, const Vec4<T1>& axis){
	T1 theta = radians(degrees);
	Vec3<T1> ax_normed = normalize(toVec3(axis));

	Mat3<T1> aux = static_cast<T1>(cos(theta)) * Mat3<T1>(one<T1>()) + 
			(one<T1>()-static_cast<T1>(cos(theta))) * outerProd(ax_normed, ax_normed) + 
			static_cast<T1>(sin(theta)) * starMat(ax_normed);
	Mat4<T1> ret(one<T1>());
	for(int i = 0; i<3; i++)
		for(int j=0; j<3; j++)
			ret[i][j] = aux[i][j];
	return ret;
}

template <typename T1>
Mat4<T1> scaleMat(const T1 &sx, const T1 &sy, const T1 &sz){
  return Mat4<T1>(sx, sy, sz, one<T1>());
}

template <typename T1>
Mat4<T1> translateMat(const T1 &tx, const T1 &ty, const T1 &tz){
  Mat4<T1> ret(one<T1>());
	ret[0][3] = tx;
	ret[1][3] = ty;
	ret[2][3] = tz;
  return ret;
}

};
