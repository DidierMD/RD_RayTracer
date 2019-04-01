// Version 1

#ifndef DOBLEFOR
#define DOBLEFOR

#include <iostream>
#include <iomanip>
#include <mutex>

// Clase útil para gestionar el modo en que distintos hilos se reparten una matriz
class DobleFor{
   private:
      int I,J;
      int InfJ, SupI, SupJ; // Los límites del for. I, J, no llegan a los Sup
      std::mutex Seguro; 
   
   public:
      DobleFor(int infi, int infj, int supi, int supj){
   		I = infi;
   		InfJ = J = infj;
   		SupI = supi;
   		SupJ = supj;
		};
      bool getIndexes(int& i, int& j){ // Devuelve true mientras la obtención sea exitosa
   		std::lock_guard<std::mutex> guard(Seguro);
   		if(I >= SupI) return false;
   		i = I;
   		j = J;
   		if(++J >= SupJ){
      		J=InfJ;
      		I++;
				std::cout << std::fixed << std::setprecision(2) << (float)i * 100. / (float)SupI << " %" << std::endl;
   		}   
   		return true;
		}; 
};

#endif

/*Cambios:
*/
