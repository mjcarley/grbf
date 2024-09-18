## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src -I$(includedir)
LDADD = $(top_builddir)/src/.libs/libgrbf.a 

DEPS = $(top_builddir)/src/libgrbf.la

EXTRA_DIST = grbf-test.c.in

bin_PROGRAMS = \
	grbf-test \
	grbf-test-float

BUILT_SOURCES = \
	grbf-test.c \
	grbf-test-float.c

CLEANFILES = \
	grbf-test.c \
	grbf-test-float.c

grbf-test.c: grbf-test.c.in
	./make-source grbf-test

grbf-test-float.c: grbf-test.c.in
	./make-source-float grbf-test

