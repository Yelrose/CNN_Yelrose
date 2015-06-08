#ifndef DEBUG_H
#define DEBUG_H

#define LITER_TO_STRING_INTER(x) #x
#define LITER_TO_STRING(x) LITER_TO_STRING_INTER(x)
#define INFO __FILE__": "LITER_TO_STRING(__LINE__)" "

#endif
