// wave8.h

#define WIDTH 768
#define LENGTH 768
#define LWS 8
#define SCALE (4.0f)
#define DIRECTIONS 5
#define SIZE (WIDTH*LENGTH*DIRECTIONS)
// We're rendering with triangle strips using glMultiDrawArrays. Thus each
// row of vertices, except for the first and the last, must be replicated
// to serve as the bottom of one row and the top of the next.
#define VCOUNT (2*WIDTH*(LENGTH-1))
#define COLOR_OFF (3*VCOUNT)

// Here we reorder storage from the natural, [LENGTH][WIDTH][DIRECTIONS],
// to one that works better with memory accesses, [LENGTH][DIRECTIONS][WIDTH].
#define store(i,j,k) ((i)*(WIDTH*DIRECTIONS)+(k)*WIDTH+(j))
#define dist(i,j,k) dist[store(i,j,k)]


