AC_INIT([grbf],1.0.0)
AC_CONFIG_SRCDIR([src/grbf.h])

GRBF_MAJOR_VERSION=1
GRBF_MINOR_VERSION=0
GRBF_MICRO_VERSION=0
GRBF_INTERFACE_AGE=0
GRBF_BINARY_AGE=0
GRBF_VERSION=$GRBF_MAJOR_VERSION.$GRBF_MINOR_VERSION.$GRBF_MICRO_VERSION

AC_SUBST(GRBF_MAJOR_VERSION)
AC_SUBST(GRBF_MINOR_VERSION)
AC_SUBST(GRBF_MICRO_VERSION)
AC_SUBST(GRBF_VERSION)

# libtool versioning
LT_RELEASE=$GRBF_MAJOR_VERSION.$GRBF_MINOR_VERSION
LT_CURRENT=`expr $GRBF_MICRO_VERSION - $GRBF_INTERFACE_AGE`
LT_REVISION=$GRBF_INTERFACE_AGE
LT_AGE=`expr $GRBF_BINARY_AGE - $GRBF_INTERFACE_AGE`
AC_SUBST(LT_RELEASE)
AC_SUBST(LT_CURRENT)
AC_SUBST(LT_REVISION)
AC_SUBST(LT_AGE)

# For automake.
VERSION=$GRBF_VERSION
PACKAGE=grbf

AC_SUBST(PACKAGE)
AC_SUBST(VERSION)

AM_INIT_AUTOMAKE(subdir-objects)

##$PACKAGE, $VERSION, no-define)

# Specify a configuration file
AM_CONFIG_HEADER(config.h)

AC_DEFINE_UNQUOTED(GRBF_MAJOR_VERSION, $GRBF_MAJOR_VERSION, [Major version])
AC_DEFINE_UNQUOTED(GRBF_MINOR_VERSION, $GRBF_MINOR_VERSION, [Minor version])
AC_DEFINE_UNQUOTED(GRBF_MICRO_VERSION, $GRBF_MICRO_VERSION, [Micro version])
AC_DEFINE_UNQUOTED(GRBF_INTERFACE_AGE, $GRBF_INTERFACE_AGE, [Interface age])
AC_DEFINE_UNQUOTED(GRBF_BINARY_AGE, $GRBF_BINARY_AGE, [Binary age])

dnl Initialize libtool
AM_PROG_LIBTOOL

dnl Initialize maintainer mode
#AM_MAINTAINER_MODE

AC_PROG_CC
AM_PROG_CC_C_O

if test x$GCC = xyes ; then
  CFLAGS="$CFLAGS -Wall -Werror-implicit-function-declaration -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations"
fi

AC_PROG_AWK
AC_ISC_POSIX
AC_HEADER_STDC


AC_CHECK_PROG(BLASWRAP, blaswrap-config, yes, no)
if test x$BLASWRAP = xyes ; then
   AC_DEFINE(HAVE_BLASWRAP, 1, [BLAS wrapper header available])
   blaswrap_cflags=`blaswrap-config --cflags`
   blaswrap_libs=`blaswrap-config --libs`
else
  echo \#################################################################
  echo \#
  echo \# BLAS wrapper library not installed or not properly installed.
  echo \# You can get it from
  echo \#
  echo \# https://github.com/mjcarley/blaswrap
  echo \#
  echo \#################################################################
  exit
fi

PKG_CHECK_MODULES([GLIB], [glib-2.0], [],
[
  echo \#################################################################
  echo \# 
  echo \# Required package GLIB-2.0 not installed. You can install it
  echo \# using your distribution's package manager or from:
  echo \#
  echo \# https://gitlab.gnome.org/GNOME/glib/
  echo \#
  echo \#################################################################
  exit
])

PKG_CHECK_MODULES([FFTW], [fftw3], [
   AC_DEFINE(HAVE_FFTW3, 1, [FFTW3 available])
], [
  echo \#################################################################
  echo \# 
  echo \# Required package FFTW3 not installed. You can install it using
  echo \# your distribution's package manager or from:
  echo \#
  echo \# http://fftw.org/download.html
  echo \#
  echo \#################################################################
  exit
])
PKG_CHECK_MODULES([FFTWF], [fftw3f], [
   AC_DEFINE(HAVE_FFTW3F, 1, [FFTW3F available])
], [
  echo \#################################################################
  echo \# 
  echo \# Required package FFTW3 not installed. You can install it using
  echo \# your distribution's package manager or from:
  echo \#
  echo \# http://fftw.org/download.html
  echo \#
  echo \#################################################################
  exit
])

AX_GCC_X86_CPU_SUPPORTS(avx, hasavx=yes) 
AX_GCC_X86_CPU_SUPPORTS(avx2, hasavx2=yes) 
AX_GCC_X86_CPU_SUPPORTS(fma, hasfma=yes) 

if test x$hasavx = xyes; then
   SIMD_FLAGS="$SIMD_FLAGS -mavx -DWBFMM_USE_AVX"
fi
if test x$hasfma = xyes; then
   SIMD_FLAGS="$SIMD_FLAGS -mfma -DWBFMM_USE_AVX"
fi

AC_CHECK_LIB(m, cos)
AC_CHECK_LIB(matheval, evaluator_evaluate)
AC_CONFIG_MACRO_DIR([m4])

AC_OPENMP
AC_PROG_F77
AC_F77_LIBRARY_LDFLAGS
AC_F77_DUMMY_MAIN
AC_F77_WRAPPERS

#DX_DOXYGEN_FEATURE(ON)
##DX_DOT_FEATURE(ON)
#DX_HTML_FEATURE(ON)
##DX_CHM_FEATURE(OFF)
##DX_CHI_FEATURE(OFF)
##DX_MAN_FEATURE(OFF)
##DX_RTF_FEATURE(OFF)
##DX_XML_FEATURE(OFF)
##DX_PDF_FEATURE(OFF)
##DX_PS_FEATURE(OFF)
#DX_INIT_DOXYGEN(grbf, doc/grbf.dxy)

CPPFLAGS="$glib_cflags $blaswrap_cflags $fftw_cflags"
CFLAGS="$CFLAGS $OPENMP_CFLAGS $SIMD_FLAGS $FFTW_CFLAGS $FFTW_CFLAGS $GLIB_CFLAGS $blaswrap_cflags"
LIBS="$GLIB_LIBS $FFTW_LIBS $FFTWF_LIBS $blaswrap_libs $LIBS"

AC_SUBST(CPPFLAGS)
AC_SUBST(CFLAGS)
AC_SUBST(LDFLAGS)
AC_SUBST(XFOIL)

LIBS="$glib_libs $LIBS $FLIBS"

AC_CONFIG_FILES([
Makefile
grbf.pc
src/Makefile
tools/Makefile])

AC_OUTPUT

