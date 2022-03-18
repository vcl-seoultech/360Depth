#ifndef T_TYPE
#define T_TYPE

#define dPair std::pair<double, double>
#define iPair std::pair<int, int>
#define INF (double)1e9

typedef struct _img_int  {
    int row, col;
} img;

typedef struct _image_double {
    double row, col;
} img_d;

typedef struct _equirectangular {
    double phi, theta;
} Equi;

typedef struct _spherical {
    double rho, phi, theta;
} Spherical;

typedef struct _cartesian {
    double x, y, z;
} Cartesian;

namespace DEMO
{

enum class Pixel { FIX, MEAN, CLOSEST };

} //namespace DEMO

#endif // T_TYPE