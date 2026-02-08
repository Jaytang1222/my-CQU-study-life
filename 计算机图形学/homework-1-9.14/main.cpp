#include<iostream>
#include<cmath>
using namespace std;

const double pi = 3.14159265358979323846;

int main() {
    double x = 2, y = 1;
    double angle = 45.0 * pi / 180.0;
    double resx = x * cos(angle) - y * sin(angle) + 1;
    double resy = x * sin(angle) + y * cos(angle) + 2;
    cout << "New coordinates = " << resx << " , " << resy;
    return 0;
}