AM_CPPFLAGS = -I$(top_srcdir) -I$(includedir) -DG_LOG_DOMAIN=\"GRBF\"

lib_LTLIBRARIES = libgrbf.la

libgrbf_la_LDFLAGS = -version-info $(LT_CURRENT):$(LT_REVISION):$(LT_AGE)\
	-release $(LT_RELEASE) -export-dynamic

libgrbf_la_SOURCES = \
	grbf-private.h \
	grbf.c \
	convolve.c \
	cardinal.c

include_HEADERS = \
	grbf.h

CLEANFILES = $(BUILT_SOURCES)
