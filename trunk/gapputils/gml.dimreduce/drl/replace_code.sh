#!/bin/tcsh

set files = `find ./ -name '*.h' | grep -v gsl_cblas ; find ./ -name '*.cpp' | grep -v gsl_cblas ; find ../mathutils/ -name '*.cpp' ; find ../ya_test/ -name '*.cpp'`
rm -rf /tmp/cpp5678
mkdir /tmp/cpp5678
mkdir /tmp/cpp5678/dimred

foreach file ( $files )
	/bin/cp $file /tmp/cpp5678/$file:t:t
	# ------ Sed Replace
	sed 's/VM_Transpose/YA_TransT/g' /tmp/cpp5678/$file:t:t > $file
	# ------ Awk insert matrix type info
	#awk 'BEGIN{info=0}info==1 && $1=="VME_MOD_BLAS"{print "    VME_BLAS_STRIDE = VME_BLAS,"}{print $0}$2=="MatrixInfo"{info=1}info==1 && $1=="};"{info=0}' /tmp/cpp5678/$file:t:t > $file
	# ------ Awk remove matrix type info
	#awk 'BEGIN{info=0}info==1 && $1!="VME_USE_MODB"{print $0}info!=1{print $0}$2=="MatrixInfo"{info=1}info==1 && $1=="};"{info=0}' /tmp/cpp5678/$file:t:t > $file
end
