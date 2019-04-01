
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

///////////Scene

template <typename T>
bool Scene<T>::intersect(const Ray<T>& ray_, Intersection<T>& best_hit_info) const{
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

template <typename T>
Scene<T>::Scene(void){
}

template <typename T>
void Scene<T>::add(const DirectionalLight<T>& light_){
	DirectionalLights.push_back(light_);
}

template <typename T>
void Scene<T>::add(const PointLight<T>& light_){
	PointLights.push_back(light_);
}

template <typename T>
void Scene<T>::add(const Triangle<T>& tri_){
	Triangles.push_back(tri_);
}

template <typename T>
void Scene<T>::add(const Sphere<T>& sph_){
	Spheres.push_back(sph_);
}

template <typename T>
void Scene<T>::createStructures(void){
	LightsPtrs.clear();
	int tam = DirectionalLights.size();
	for(int i = 0; i<tam; i++)
		LightsPtrs.push_back(&DirectionalLights[i]);

	tam = PointLights.size();
	for(int i = 0; i<tam; i++)
		LightsPtrs.push_back(&PointLights[i]);

	ObjectsPtrs.clear();
	tam = Triangles.size();
	for(int i = 0; i<tam; i++)
		ObjectsPtrs.push_back(&Triangles[i]);

	tam = Spheres.size();
	for(int i = 0; i<tam; i++)
		ObjectsPtrs.push_back(&Spheres[i]);
}

template <typename T>
Vec3<T> Scene<T>::computeLight(const Vec3<T>& eye_direction, const Vec3<T>& light_direction, 
                         		 const Object<T>* hit_obj, const Vec3<T>& normal_, const Light<T>* light_)const{
	Vec3<T> half_vec = myml::normalize(eye_direction + light_direction);
	
	Vec3<T> lambert = std::max(myml::dotProd(normal_, light_direction), zero<T>()) * hit_obj->diffuse();
	Vec3<T> phong = static_cast<T>(pow(std::max(myml::dotProd(normal_, half_vec), zero<T>()), hit_obj->shininess())) *
						 hit_obj->specular();
	return (lambert + phong) * light_->color();
}

template <typename T>
Vec3<T> Scene<T>::calculateColor(const Ray<T>& ray_, int refl_depth_){
	Vec3<T> col;
	if(refl_depth_ < 0) return col;

	Intersection<T> hit_info;
	if(false == intersect(ray_, hit_info))
		return col;

	Vec3<T> hitpoint = ray_.alongRay(hit_info.tParam());
	Intersection<T> hit_aux;
	int tam = LightsPtrs.size();
	for(int i=0; i<tam; i++){
		Ray<T> raytolight;
		T light_t = LightsPtrs[i] -> getRay(hitpoint, raytolight);

		hit_aux.setTParam(myml::infinity<T>());
		bool hits = intersect(raytolight, hit_aux);

		if(false == hits || hit_aux.tParam() > light_t){
			col = col + LightsPtrs[i]->attenuation(hitpoint) * computeLight(-ray_.direction(), raytolight.direction(),
																hit_info.hitObject(), hit_info.normalVec(), LightsPtrs[i]);
		}
	}
	Ray<T> refl_ray = reflectionRay(hitpoint, hit_info.normalVec(), ray_.direction());
	Vec3<T> reflectioncol = hit_info.hitObject()->specular() * calculateColor(refl_ray, refl_depth_-1);
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
			else if (cmd == "output") {
				s >> Output;
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
void RayTracer<T>::saveImage(void) const{
	if(Output.size() != 0)
		ActualImage.saveToFile(Output);
	else
		ActualImage.saveToFile("RayTracedImage.png");
}


