
ALL: lib

CFLAGS   =
FFLAGS   =
ZLIBSOURCE = adler32.c compress.c crc32.c deflate.c gzclose.c gzlib.c gzread.c gzwrite.c infback.c inffast.c inflate.c inftrees.c trees.c uncompr.c zutil.c
SZSOURCE   = ByteToolkit.c CompressElement.c DynamicByteArray.c DynamicDoubleArray.c DynamicFloatArray.c DynamicIntArray.c Huffman.c TightDataPointStorageD.c TightDataPointStorageF.c TightDataPointStorageI.c TypeManager.c VarSet.c callZlib.c conf_sz.c dataCompression.c dictionary.c iniparser.c rw.c rwf.c sz.c sz_double.c sz_double_pwr.c sz_float.c sz_float_pwr.c sz_int16.c sz_int32.c sz_int64.c sz_int8.c szd_double.c szd_double_pwr.c szd_float.c szd_float_pwr.c szd_int16.c szd_int32.c szd_int64.c szd_int8.c szf.c
MD5SOURCE = md5.c
JRSOURCE = galois.c gf.c gf_cpu.c gf_w128.c gf_w16.c gf_w32.c gf_w4.c gf_w64.c gf_w8.c gf_wgen.c jerasure.c
FTISOURCE= api.c checkpoint.c conf_fti.c meta.c postckpt.c postreco.c recover.c tools.c topo.c
SOURCEC  = ${ZLIBSOURCE} ${SZSOURCE} ${MD5SOURCE} ${JRSOURCE} ${FTISOURCE} gmres.c borthog.c borthog2.c gmres2.c gmreig.c gmpre.c
ZLIBH    = crc32.h deflate.h gzguts.h inffast.h inffixed.h inflate.h inftrees.h trees.h zconf.h zlib.h zutil.h
SZH		 = ByteToolkit.h CompressElement.h DynamicByteArray.h DynamicDoubleArray.h DynamicFloatArray.h DynamicIntArray.h Huffman.h TightDataPointStorageD.h TightDataPointStorageF.h TightDataPointStorageI.h TypeManager.h VarSet.h callZlib.h conf.h dataCompression.h dictionary.h iniparser.h include/rw.h sz.h sz_double.h sz_double_pwr.h sz_float.h sz_float_pwr.h sz_int16.h sz_int32.h sz_int64.h sz_int8.h szd_double.h szd_double_pwr.h szd_float.h szd_float_pwr.h szd_int16.h szd_int32.h szd_int64.h szd_int8.h szf.h
MD5H	 = md5.h
JRH      = galois.h gf_complete.h gf_cpu.h gf_int.h gf_w16.h gf_w32.h gf_w4.h gf_w64.h gf_w8.h jerasure.h
FTIH	 = interface.h fti.h
SOURCEH  = ${ZLIBH} ${SZH} ${MD5H} ${JRH} ${FTIH} gmresimpl.h
SOURCEF  =
LIBBASE  = libpetscksp
DIRS     = lgmres fgmres dgmres pgmres pipefgmres agmres
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/gmres/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
