#linux: make clean all test
#winds: make; "C:\Python35-32\python.exe" __init__.py

all:
ifeq ($(OS),Windows_NT)
# http://www.mingw.org/wiki/sampledll
	g++ -c -DBUILDING_EXAMPLE_DLL projection.cpp
	g++ -shared -o ppolytope.lib projection.o -Wl,--out-implib,libppolytope.a
else
	g++ -shared -Wl,-soname,ppolytope -o ppolytope.so -fPIC projection.cpp -Wall
	mv $(SCRATCH)/ppolytope.so ./ppolytope.lib
endif

clean:
	rm ppolytope.lib
ifeq ($(OS),Windows_NT)
	rm projection.o libppolytope.a
endif

test:
	python __init__.py