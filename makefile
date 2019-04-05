SRC_DIR=./src
LIB_DIR=$(SRC_DIR)/lib
INCL_DIR=$(SRC_DIR)/include
BIN_DIR=./bin

LINK_DEP= -lfreeimage

$(BIN_DIR)/RD_RT: $(SRC_DIR)/Main.cpp $(INCL_DIR)/MyML.h $(INCL_DIR)/RayTracer.h $(INCL_DIR)/DobleFor.h $(INCL_DIR)/SceneDescription.h $(INCL_DIR)/AccelerationStructures.h
	c++ -std=c++11 -stdlib=libc++ -Wall -O2 -o $(BIN_DIR)/RD_RT $(SRC_DIR)/Main.cpp -I$(INCL_DIR) -L$(LIB_DIR) $(LINK_DEP)
