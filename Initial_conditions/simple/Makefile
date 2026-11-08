FC = gfortran
FFLAGS = -O2 -ffree-line-length-none

.PHONY: all clean

all: public_nse.exe run_nse.exe

public_nse.exe: public_nse.f90 implno.dek const.dek network.dek
	$(FC) $(FFLAGS) public_nse.f90 -o public_nse.exe

# run_nse.exe (single-shot, command-line driver for scripting) reuses the
# nse()/nsefunc()/init_network()/... subroutines straight out of
# public_nse.f90, stripping only its interactive `program testnse` block,
# so there is exactly one copy of the physics and it can't drift out of
# sync with public_nse.f90.
run_nse.exe: run_nse.f90 public_nse.f90 implno.dek const.dek network.dek
	awk '/^      subroutine /{f=1} f' public_nse.f90 > .nse_lib.f90
	$(FC) $(FFLAGS) run_nse.f90 .nse_lib.f90 -o run_nse.exe
	rm -f .nse_lib.f90

clean:
	rm -f public_nse.exe run_nse.exe .nse_lib.f90 *.o
