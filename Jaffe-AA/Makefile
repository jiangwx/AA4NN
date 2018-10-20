CC = g++
EXECUTABLE = Jaffa-AA
DEPS = $(shell find ./ -name "*.h")
SRC = $(shell find ./ -name "*.cpp")
IDIRS = -I/opt/OpenBLAS/include -I.
LDIRS = -L=/usr/local/lib -L=/opt/OpenBLAS/lib/
LLIBS = -lopenblas -laffa
OLEVEL = -O3
CFLAGS = ${OLEVEL} -c ${IDIRS}
LFLAGS = ${OLEVEL} ${LDIRS} ${LLIBS}

DEPS = $(shell find ./ -name "*.h")
SOURCES = $(shell find ./ -name "*.cpp")
OBJECTS = $(src:%.cpp=%.o) 

.PHONY: all
all: ${EXECUTABLE}
${EXECUTABLE}: ${OBJECTS}
	${CC} ${OBJECTS} ${LFLAGS} -o $@ -include ${DEPS}
%.o: %.cpp
	${CC} ${CFLAGS} $< -o $@
clean:
	${RM} ${EXECUTABLE} ${OBJECTS}