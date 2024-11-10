
test : utils.o
	cc -o test tests/test_utils.c src/utils.o -lcriterion -lgsl

utils.o : src/utils.h
	cc -c -o src/utils.o src/utils.c

clean :
	rm src/*.o
	rm test
