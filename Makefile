SHELL=/bin/bash
.SUFFIXES:
default:

.PHONY: create-labs

STUDENT_EDITABLE_FILES=matexp_solution.hpp config.make
PRIVATE_FILES=Assignment.key.ipynb admin .git solution bad-solution

COMPILER=$(CXX) 
MICROBENCH_OPTIMIZE= -DHAVE_LINUX_PERF_EVENT_H -I$(PWD) -g $(C_OPTS)
LIBS= -lm -pthread -lboost_program_options -L/usr/lib/ -lboost_system -ldl
BUILD=build/

OPTIMIZE+=-march=x86-64
COMPILER=g++-9
include config.make

.PHONY: autograde
autograde: matexp.exe 
	./matexp.exe -M 3700 -o bench.csv -i 10 -f bench_solution
	./matexp.exe -M 3700 -o correctness.csv -v -s 300 600 -i 1 -p 4 5 -f matexp_solution_c

.PRECIOUS: $(BUILD)%.cpp
.PRECIOUS: $(BUILD)%.hpp

$(BUILD)perfstats.o: perfstats.c perfstats.h
	mkdir -p $(BUILD) 
	cp  $< $(BUILD)$<
	$(COMPILER) -DHAVE_LINUX_PERF_EVENT_H -O3 -I$(PWD) $(LIBS) -o $(BUILD)perfstats.o -c $(BUILD)perfstats.c


$(BUILD)%.s: $(BUILD)%.cpp
	mkdir -p $(BUILD) 
#	cp  $< $(BUILD)$<
	$(COMPILER) $(MICROBENCH_OPTIMIZE) $(LIBS) -S $(BUILD)$*.cpp

$(BUILD)%.so: $(BUILD)%.cpp
	mkdir -p $(BUILD) 
	cp *.hpp $(BUILD)
	cp *.h   $(BUILD)
	$(COMPILER)  -DHAVE_LINUX_PERF_EVENT_H $(MICROBENCH_OPTIMIZE) $(LIBS) -rdynamic -fPIC -shared -o $(BUILD)$*.so $(BUILD)$*.cpp
#	$(COMPILER) $(MICROBENCH_OPTIMIZE) $(LIBS) -c -fPIC -o $(BUILD)$*.o $(BUILD)$*.cpp

$(BUILD)%.cpp: %.cpp
	cp $< $(BUILD)
	cp *.hpp $(BUILD)

        
$(BUILD)matexp.o : OPTIMIZE=$(MATEXP_OPTIMIZE)
$(BUILD)matexp.s : OPTIMIZE=$(MATEXP_OPTIMIZE)
$(BUILD)matexp_main.o : OPTIMIZE=$(MATEXP_OPTIMIZE)

matexp.exe:  $(BUILD)matexp_main.o  $(BUILD)matexp.o $(BUILD)perfstats.o
	$(COMPILER) $(MATEXP_OPTIMIZE) $(BUILD)matexp_main.o  $(BUILD)perfstats.o $(BUILD)matexp.o -o matexp.exe

$(BUILD)run_tests.o : OPTIMIZE=-O3

$(BUILD)%.o: %.cpp
	mkdir -p $(BUILD) 
	cp  $< $(BUILD)$<
	$(COMPILER)  -DHAVE_LINUX_PERF_EVENT_H $(MATEXP_OPTIMIZE) $(MICROBENCH_OPTIMIZE) $(LIBS) -o $(BUILD)$*.o -c $(BUILD)$*.cpp


fiddle.exe:  $(BUILD)fiddle.o $(FIDDLE_OBJS) $(BUILD)perfstats.o
	$(COMPILER) $(MICROBENCH_OPTIMIZE)  -DHAVE_LINUX_PERF_EVENT_H $(BUILD)fiddle.o $(BUILD)perfstats.o $(FIDDLE_OBJS) $(LIBS) -o fiddle.exe
#fiddle.exe: EXTRA_LDFLAGS=-pg
#$(BUILD)fiddle.o : OPTIMIZE=-O3 -pg


#-include $(DJR_JOB_ROOT)/$(LAB_SUBMISSION_DIR)/config.make
clean: 
	rm -f *.exe $(BUILD)*
