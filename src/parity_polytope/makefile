#Usage: make clean; make; python __init__.py

all: projection.cpp
	g++ -shared -Wl,-soname,ppolytope -o ppolytope.so -fPIC projection.cpp -Wall

clean:
	rm $(SCRATCH)/*.so
