// This code was created by pygmsh vunknown.
p0 = newp;
Point(p0) = {0.0, 0.0, 0.0, 1};
p1 = newp;
Point(p1) = {0.0, 1.0, 0.0, 1};
p2 = newp;
Point(p2) = {1.0, 1.5, 0.0, 1};
p3 = newp;
Point(p3) = {3.0, 1.5, 0.0, 1};
p4 = newp;
Point(p4) = {3.0, 4.0, 0.0, 1};
p5 = newp;
Point(p5) = {3.0, 5.0, 0.0, 1};
p6 = newp;
Point(p6) = {1.0, 5.0, 0.0, 1};
p7 = newp;
Point(p7) = {-2.0, 2.0, 0.0, 1};
p8 = newp;
Point(p8) = {-2.0, 0.0, 0.0, 1};
p9 = newp;
Point(p9) = {-1.0, 0.0, 0.0, 1};
l0 = newl;
BSpline(l0) = {p0, p1, p2, p3};
l1 = newl;
BSpline(l1) = {p3, p4, p5};
l2 = newl;
BSpline(l2) = {p5, p6, p7, p8};
l3 = newl;
BSpline(l3) = {p8, p9, p0};
ll0 = newll;
Line Loop(ll0) = {l0, l1, l2, l3};
s0 = news;
Plane Surface(s0) = {ll0};
Physical Surface("biquadratic") = {s0};
Physical Line("bot_bc") = {l0};
Physical Line("top_bc") = {l1};