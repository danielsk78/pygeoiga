// This code was created by pygmsh vunknown.
SetFactory("OpenCASCADE");
p0 = newp;
Point(p0) = {0.0, -10.0, 0.0, 10};
p1 = newp;
Point(p1) = {0.0, 0.0, 0.0, 10};
p2 = newp;
Point(p2) = {0.0, 10.0, 0.0, 10};
p3 = newp;
Point(p3) = {50.0, 10.0, 0.0, 10};
p4 = newp;
Point(p4) = {100.0, 10.0, 0.0, 10};
p5 = newp;
Point(p5) = {100.0, 0.0, 0.0, 10};
p6 = newp;
Point(p6) = {100.0, -10.0, 0.0, 10};
p7 = newp;
Point(p7) = {50.0, -10.0, 0.0, 10};
l0 = newl;
BSpline(l0) = {p0, p1, p2};
l1 = newl;
BSpline(l1) = {p2, p3, p4};
l2 = newl;
BSpline(l2) = {p4, p5, p6};
l3 = newl;
BSpline(l3) = {p6, p7, p0};
ll0 = newll;
Line Loop(ll0) = {l0, l1, l2, l3};
s0 = news;
Plane Surface(s0) = {ll0};
Physical Surface("patch") = {s0};
Physical Line("bot_bc") = {l0};
Physical Line("right_bc") = {l1};
Physical Line("top_bc") = {l2};
Physical Line("left_bc") = {l3};