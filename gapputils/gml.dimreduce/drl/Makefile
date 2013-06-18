#***************************************************************************
#                                  Makefile 
#                             -------------------
#
#  Build for general CPP utils
#
#  __________________________________________________________________________
#
#    begin                : Thu Oct 9 2003
#    copyright            : (C) 2003 by W. Michael Brown
#    email                : wmbrown@sandia.gov
#***************************************************************************/

include ../Configuration.mk
OBJ_DIR  = $(HOBJ_DIR)/math

#Objects which are compiled here
OBJS            = 

all:

gslcblas:
	cd cblas; make clean; make; cd ../

gslcblasclean:
	cd cblas; make clean; cd ../

clean:
	rm -f *.o a.out core

veryclean: clean
	rm -f *~ ./api
