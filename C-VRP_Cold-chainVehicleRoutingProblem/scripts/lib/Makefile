.DEFAULT_GOAL := all
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
SUFFIX := $(shell python3-config --extension-suffix)

all:
	clang++ -O3 -Wall -shared -std=c++17 -fPIC \
	$(PYBIND11_INCLUDES) cvrptw_bindings.cpp cvrptw_ops.cpp \
	-o cvrptw_ops$(SUFFIX)

clean:
	rm -f *.so
