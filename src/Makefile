exe = fado

cc = "$(shell which nvcc)" 
flags = -I. -O3 -Xptxas -dlcm=ca

ifeq ($(debug), 1)
	flags+= -DDEBUG 
endif

objs = $(patsubst %.cu,%.o,$(wildcard *.cu))

deps = $(wildcard ./*.cuh) \
		$(wildcard *.h) \
		Makefile

%.o:%.cu $(deps)
	$(cc) -c $< -o $@ $(flags)

$(exe):$(objs)
	$(cc) $(objs) -o $(exe) $(flags)

clean:
	rm -rf $(exe) $(objs) 
