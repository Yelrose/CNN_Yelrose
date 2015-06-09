object =  include/*.h
test: $(object)
	g++ -o bin/test -I include/ src/test.cpp

