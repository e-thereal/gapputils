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

include ../../Configuration.mk
OBJ_DIR = ../$(HOBJ_DIR)/math/cblas
CBLASLIB = ../$(HOBJ_DIR)/libcblas.a

OBJECTS = $(OBJ_DIR)/sasum.o $(OBJ_DIR)/saxpy.o $(OBJ_DIR)/scasum.o $(OBJ_DIR)/scnrm2.o \
 $(OBJ_DIR)/scopy.o $(OBJ_DIR)/sdot.o $(OBJ_DIR)/sdsdot.o $(OBJ_DIR)/sgbmv.o $(OBJ_DIR)/sgemm.o $(OBJ_DIR)/sgemv.o $(OBJ_DIR)/sger.o \
 $(OBJ_DIR)/snrm2.o $(OBJ_DIR)/srot.o $(OBJ_DIR)/srotg.o $(OBJ_DIR)/srotm.o $(OBJ_DIR)/srotmg.o $(OBJ_DIR)/ssbmv.o $(OBJ_DIR)/sscal.o \
 $(OBJ_DIR)/sspmv.o $(OBJ_DIR)/sspr.o $(OBJ_DIR)/sspr2.o $(OBJ_DIR)/sswap.o $(OBJ_DIR)/ssymm.o $(OBJ_DIR)/ssymv.o $(OBJ_DIR)/ssyr.o \
 $(OBJ_DIR)/ssyr2.o $(OBJ_DIR)/ssyr2k.o $(OBJ_DIR)/ssyrk.o $(OBJ_DIR)/stbmv.o $(OBJ_DIR)/stbsv.o $(OBJ_DIR)/stpmv.o \
 $(OBJ_DIR)/stpsv.o $(OBJ_DIR)/strmm.o $(OBJ_DIR)/strmv.o $(OBJ_DIR)/strsm.o $(OBJ_DIR)/strsv.o $(OBJ_DIR)/dasum.o $(OBJ_DIR)/daxpy.o \
 $(OBJ_DIR)/dcopy.o $(OBJ_DIR)/ddot.o $(OBJ_DIR)/dgbmv.o $(OBJ_DIR)/dgemm.o $(OBJ_DIR)/dgemv.o $(OBJ_DIR)/dger.o $(OBJ_DIR)/dnrm2.o \
 $(OBJ_DIR)/drot.o $(OBJ_DIR)/drotg.o $(OBJ_DIR)/drotm.o $(OBJ_DIR)/drotmg.o $(OBJ_DIR)/dsbmv.o $(OBJ_DIR)/dscal.o $(OBJ_DIR)/dsdot.o \
 $(OBJ_DIR)/dspmv.o $(OBJ_DIR)/dspr.o $(OBJ_DIR)/dspr2.o $(OBJ_DIR)/dswap.o $(OBJ_DIR)/dsymm.o $(OBJ_DIR)/dsymv.o $(OBJ_DIR)/dsyr.o \
 $(OBJ_DIR)/dsyr2.o $(OBJ_DIR)/dsyr2k.o $(OBJ_DIR)/dsyrk.o $(OBJ_DIR)/dtbmv.o $(OBJ_DIR)/dtbsv.o $(OBJ_DIR)/dtpmv.o \
 $(OBJ_DIR)/dtpsv.o $(OBJ_DIR)/dtrmm.o $(OBJ_DIR)/dtrmv.o $(OBJ_DIR)/dtrsm.o $(OBJ_DIR)/dtrsv.o $(OBJ_DIR)/dzasum.o \
 $(OBJ_DIR)/dznrm2.o $(OBJ_DIR)/caxpy.o $(OBJ_DIR)/ccopy.o $(OBJ_DIR)/cdotc_sub.o $(OBJ_DIR)/cdotu_sub.o $(OBJ_DIR)/cgbmv.o \
 $(OBJ_DIR)/cgemm.o $(OBJ_DIR)/cgemv.o $(OBJ_DIR)/cgerc.o $(OBJ_DIR)/cgeru.o $(OBJ_DIR)/chbmv.o $(OBJ_DIR)/chemm.o $(OBJ_DIR)/chemv.o \
 $(OBJ_DIR)/cher.o $(OBJ_DIR)/cher2.o $(OBJ_DIR)/cher2k.o $(OBJ_DIR)/cherk.o $(OBJ_DIR)/chpmv.o $(OBJ_DIR)/chpr.o $(OBJ_DIR)/chpr2.o \
 $(OBJ_DIR)/cscal.o $(OBJ_DIR)/csscal.o $(OBJ_DIR)/cswap.o $(OBJ_DIR)/csymm.o $(OBJ_DIR)/csyr2k.o $(OBJ_DIR)/csyrk.o \
 $(OBJ_DIR)/ctbmv.o $(OBJ_DIR)/ctbsv.o $(OBJ_DIR)/ctpmv.o $(OBJ_DIR)/ctpsv.o $(OBJ_DIR)/ctrmm.o $(OBJ_DIR)/ctrmv.o $(OBJ_DIR)/ctrsm.o \
 $(OBJ_DIR)/ctrsv.o $(OBJ_DIR)/zaxpy.o $(OBJ_DIR)/zcopy.o $(OBJ_DIR)/zdotc_sub.o $(OBJ_DIR)/zdotu_sub.o $(OBJ_DIR)/zdscal.o \
 $(OBJ_DIR)/zgbmv.o $(OBJ_DIR)/zgemm.o $(OBJ_DIR)/zgemv.o $(OBJ_DIR)/zgerc.o $(OBJ_DIR)/zgeru.o $(OBJ_DIR)/zhbmv.o $(OBJ_DIR)/zhemm.o \
 $(OBJ_DIR)/zhemv.o $(OBJ_DIR)/zher.o $(OBJ_DIR)/zher2.o $(OBJ_DIR)/zher2k.o $(OBJ_DIR)/zherk.o $(OBJ_DIR)/zhpmv.o $(OBJ_DIR)/zhpr.o \
 $(OBJ_DIR)/zhpr2.o $(OBJ_DIR)/zscal.o $(OBJ_DIR)/zswap.o $(OBJ_DIR)/zsymm.o $(OBJ_DIR)/zsyr2k.o $(OBJ_DIR)/zsyrk.o \
 $(OBJ_DIR)/ztbmv.o $(OBJ_DIR)/ztbsv.o $(OBJ_DIR)/ztpmv.o $(OBJ_DIR)/ztpsv.o $(OBJ_DIR)/ztrmm.o $(OBJ_DIR)/ztrmv.o $(OBJ_DIR)/ztrsm.o \
 $(OBJ_DIR)/ztrsv.o $(OBJ_DIR)/icamax.o $(OBJ_DIR)/idamax.o $(OBJ_DIR)/isamax.o $(OBJ_DIR)/izamax.o $(OBJ_DIR)/xerbla.o

SOURCES = sasum.c saxpy.c scasum.c scnrm2.c scopy.c \
 sdot.c sdsdot.c sgbmv.c sgemm.c sgemv.c sger.c snrm2.c srot.c \
 srotg.c srotm.c srotmg.c ssbmv.c sscal.c sspmv.c sspr.c \
 sspr2.c sswap.c ssymm.c ssymv.c ssyr.c ssyr2.c ssyr2k.c \
 ssyrk.c stbmv.c stbsv.c stpmv.c stpsv.c strmm.c strmv.c \
 strsm.c strsv.c dasum.c daxpy.c dcopy.c ddot.c dgbmv.c dgemm.c \
 dgemv.c dger.c dnrm2.c drot.c drotg.c drotm.c drotmg.c dsbmv.c \
 dscal.c dsdot.c dspmv.c dspr.c dspr2.c dswap.c dsymm.c dsymv.c \
 dsyr.c dsyr2.c dsyr2k.c dsyrk.c dtbmv.c dtbsv.c dtpmv.c \
 dtpsv.c dtrmm.c dtrmv.c dtrsm.c dtrsv.c dzasum.c dznrm2.c \
 caxpy.c ccopy.c cdotc_sub.c cdotu_sub.c cgbmv.c cgemm.c \
 cgemv.c cgerc.c cgeru.c chbmv.c chemm.c chemv.c cher.c cher2.c \
 cher2k.c cherk.c chpmv.c chpr.c chpr2.c cscal.c csscal.c \
 cswap.c csymm.c csyr2k.c csyrk.c ctbmv.c ctbsv.c ctpmv.c \
 ctpsv.c ctrmm.c ctrmv.c ctrsm.c ctrsv.c zaxpy.c zcopy.c \
 zdotc_sub.c zdotu_sub.c zdscal.c zgbmv.c zgemm.c zgemv.c \
 zgerc.c zgeru.c zhbmv.c zhemm.c zhemv.c zher.c zher2.c \
 zher2k.c zherk.c zhpmv.c zhpr.c zhpr2.c zscal.c zswap.c \
 zsymm.c zsyr2k.c zsyrk.c ztbmv.c ztbsv.c ztpmv.c ztpsv.c \
 ztrmm.c ztrmv.c ztrsm.c ztrsv.c icamax.c idamax.c isamax.c \
 izamax.c xerbla.c

HEADERS = tests.c tests.h cblas.h source_asum_c.h \
 source_asum_r.h source_axpy_c.h source_axpy_r.h \
 source_copy_c.h source_copy_r.h source_dot_c.h source_dot_r.h \
 source_gbmv_c.h source_gbmv_r.h source_gemm_c.h \
 source_gemm_r.h source_gemv_c.h source_gemv_r.h source_ger.h \
 source_gerc.h source_geru.h source_hbmv.h source_hemm.h \
 source_hemv.h source_her.h source_her2.h source_her2k.h \
 source_herk.h source_hpmv.h source_hpr.h source_hpr2.h \
 source_iamax_c.h source_iamax_r.h source_nrm2_c.h \
 source_nrm2_r.h source_rot.h source_rotg.h source_rotm.h \
 source_rotmg.h source_sbmv.h source_scal_c.h source_scal_c_s.h \
 source_scal_r.h source_spmv.h source_spr.h source_spr2.h \
 source_swap_c.h source_swap_r.h source_symm_c.h \
 source_symm_r.h source_symv.h source_syr.h source_syr2.h \
 source_syr2k_c.h source_syr2k_r.h source_syrk_c.h \
 source_syrk_r.h source_tbmv_c.h source_tbmv_r.h \
 source_tbsv_c.h source_tbsv_r.h source_tpmv_c.h \
 source_tpmv_r.h source_tpsv_c.h source_tpsv_r.h \
 source_trmm_c.h source_trmm_r.h source_trmv_c.h \
 source_trmv_r.h source_trsm_c.h source_trsm_r.h \
 source_trsv_c.h source_trsv_r.h hypot.c

all: $(CBLASLIB)


$(OBJ_DIR)/%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $*.c

$(CBLASLIB): $(OBJECTS)
	$(AR) -crusv $(CBLASLIB) $(OBJECTS)

clean:
	rm -rf $(CBLASLIB) $(OBJECTS)

veryclean: clean
	rm -rf ~ 
