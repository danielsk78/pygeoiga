// This code was created by pygmsh vunknown.
p0 = newp;
Point(p0) = {0.0, 0.0, 0.0, 1};
p1 = newp;
Point(p1) = {25.0, 0.0, 0.0, 1};
p2 = newp;
Point(p2) = {75.0, 0.0, 0.0, 1};
p3 = newp;
Point(p3) = {125.00000000000001, 0.0, 0.0, 1};
p4 = newp;
Point(p4) = {175.0, 0.0, 0.0, 1};
p5 = newp;
Point(p5) = {225.0, 0.0, 0.0, 1};
p6 = newp;
Point(p6) = {275.0, 0.0, 0.0, 1};
p7 = newp;
Point(p7) = {325.0, 0.0, 0.0, 1};
p8 = newp;
Point(p8) = {375.0, 0.0, 0.0, 1};
p9 = newp;
Point(p9) = {425.0, 0.0, 0.0, 1};
p10 = newp;
Point(p10) = {475.0, 0.0, 0.0, 1};
p11 = newp;
Point(p11) = {500.0, 0.0, 0.0, 1};
p12 = newp;
Point(p12) = {500.0, 5.0, 0.0, 1};
p13 = newp;
Point(p13) = {500.0, 15.0, 0.0, 1};
p14 = newp;
Point(p14) = {500.0, 25.0, 0.0, 1};
p15 = newp;
Point(p15) = {500.0, 35.0, 0.0, 1};
p16 = newp;
Point(p16) = {500.00000000000006, 45.0, 0.0, 1};
p17 = newp;
Point(p17) = {500.00000000000006, 55.0, 0.0, 1};
p18 = newp;
Point(p18) = {500.0, 65.0, 0.0, 1};
p19 = newp;
Point(p19) = {500.0, 75.0, 0.0, 1};
p20 = newp;
Point(p20) = {500.0, 85.0, 0.0, 1};
p21 = newp;
Point(p21) = {500.0, 95.0, 0.0, 1};
p22 = newp;
Point(p22) = {500.0, 100.0, 0.0, 1};
p23 = newp;
Point(p23) = {475.0, 115.0, 0.0, 1};
p24 = newp;
Point(p24) = {425.0, 139.0, 0.0, 1};
p25 = newp;
Point(p25) = {375.0, 157.0, 0.0, 1};
p26 = newp;
Point(p26) = {325.0, 169.0, 0.0, 1};
p27 = newp;
Point(p27) = {275.0, 175.0, 0.0, 1};
p28 = newp;
Point(p28) = {225.0, 175.0, 0.0, 1};
p29 = newp;
Point(p29) = {175.0, 169.0, 0.0, 1};
p30 = newp;
Point(p30) = {125.00000000000001, 157.0, 0.0, 1};
p31 = newp;
Point(p31) = {75.0, 139.0, 0.0, 1};
p32 = newp;
Point(p32) = {25.0, 115.0, 0.0, 1};
p33 = newp;
Point(p33) = {0.0, 100.0, 0.0, 1};
p34 = newp;
Point(p34) = {0.0, 95.0, 0.0, 1};
p35 = newp;
Point(p35) = {0.0, 85.0, 0.0, 1};
p36 = newp;
Point(p36) = {0.0, 75.0, 0.0, 1};
p37 = newp;
Point(p37) = {0.0, 65.0, 0.0, 1};
p38 = newp;
Point(p38) = {0.0, 55.0, 0.0, 1};
p39 = newp;
Point(p39) = {0.0, 45.0, 0.0, 1};
p40 = newp;
Point(p40) = {0.0, 35.0, 0.0, 1};
p41 = newp;
Point(p41) = {0.0, 25.0, 0.0, 1};
p42 = newp;
Point(p42) = {0.0, 15.0, 0.0, 1};
p43 = newp;
Point(p43) = {0.0, 5.0, 0.0, 1};
l0 = newl;
BSpline(l0) = {p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11};
l1 = newl;
BSpline(l1) = {p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22};
l2 = newl;
BSpline(l2) = {p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33};
l3 = newl;
BSpline(l3) = {p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43, p0};
ll0 = newll;
Line Loop(ll0) = {l0, l1, l2, l3};
s0 = news;
Plane Surface(s0) = {ll0};
Physical Surface("granite") = {s0};
p44 = newp;
Point(p44) = {500.0, 110.0, 0.0, 1};
p45 = newp;
Point(p45) = {500.0, 130.0, 0.0, 1};
p46 = newp;
Point(p46) = {500.0, 150.0, 0.0, 1};
p47 = newp;
Point(p47) = {500.0, 170.0, 0.0, 1};
p48 = newp;
Point(p48) = {500.00000000000006, 190.0, 0.0, 1};
p49 = newp;
Point(p49) = {500.00000000000006, 210.0, 0.0, 1};
p50 = newp;
Point(p50) = {500.0, 230.0, 0.0, 1};
p51 = newp;
Point(p51) = {500.0, 250.0, 0.0, 1};
p52 = newp;
Point(p52) = {500.0, 270.0, 0.0, 1};
p53 = newp;
Point(p53) = {500.0, 290.0, 0.0, 1};
p54 = newp;
Point(p54) = {500.0, 300.0, 0.0, 1};
l4 = newl;
BSpline(l4) = {p22, p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54};
p55 = newp;
Point(p55) = {475.0, 310.0, 0.0, 1};
p56 = newp;
Point(p56) = {425.0, 326.0, 0.0, 1};
p57 = newp;
Point(p57) = {375.0, 337.99999999999994, 0.0, 1};
p58 = newp;
Point(p58) = {325.0, 345.99999999999994, 0.0, 1};
p59 = newp;
Point(p59) = {275.0, 349.99999999999994, 0.0, 1};
p60 = newp;
Point(p60) = {225.0, 349.99999999999994, 0.0, 1};
p61 = newp;
Point(p61) = {175.0, 345.99999999999994, 0.0, 1};
p62 = newp;
Point(p62) = {125.00000000000001, 337.99999999999994, 0.0, 1};
p63 = newp;
Point(p63) = {75.0, 326.0, 0.0, 1};
p64 = newp;
Point(p64) = {25.0, 310.0, 0.0, 1};
p65 = newp;
Point(p65) = {0.0, 300.0, 0.0, 1};
l5 = newl;
BSpline(l5) = {p54, p55, p56, p57, p58, p59, p60, p61, p62, p63, p64, p65};
p66 = newp;
Point(p66) = {0.0, 290.0, 0.0, 1};
p67 = newp;
Point(p67) = {0.0, 270.0, 0.0, 1};
p68 = newp;
Point(p68) = {0.0, 250.0, 0.0, 1};
p69 = newp;
Point(p69) = {0.0, 230.0, 0.0, 1};
p70 = newp;
Point(p70) = {0.0, 210.0, 0.0, 1};
p71 = newp;
Point(p71) = {0.0, 190.0, 0.0, 1};
p72 = newp;
Point(p72) = {0.0, 170.0, 0.0, 1};
p73 = newp;
Point(p73) = {0.0, 150.0, 0.0, 1};
p74 = newp;
Point(p74) = {0.0, 130.0, 0.0, 1};
p75 = newp;
Point(p75) = {0.0, 110.0, 0.0, 1};
l6 = newl;
BSpline(l6) = {p65, p66, p67, p68, p69, p70, p71, p72, p73, p74, p75, p33};
ll1 = newll;
Line Loop(ll1) = {-l2, l4, l5, l6};
s1 = news;
Plane Surface(s1) = {ll1};
Physical Surface("mudstone") = {s1};
p76 = newp;
Point(p76) = {500.0, 310.0, 0.0, 1};
p77 = newp;
Point(p77) = {500.0, 330.0, 0.0, 1};
p78 = newp;
Point(p78) = {500.0, 350.0, 0.0, 1};
p79 = newp;
Point(p79) = {500.0, 370.0, 0.0, 1};
p80 = newp;
Point(p80) = {500.00000000000006, 390.0, 0.0, 1};
p81 = newp;
Point(p81) = {500.00000000000006, 410.0, 0.0, 1};
p82 = newp;
Point(p82) = {500.0, 430.0, 0.0, 1};
p83 = newp;
Point(p83) = {500.0, 450.0, 0.0, 1};
p84 = newp;
Point(p84) = {500.0, 470.0, 0.0, 1};
p85 = newp;
Point(p85) = {500.0, 490.0, 0.0, 1};
p86 = newp;
Point(p86) = {500.0, 500.0, 0.0, 1};
l7 = newl;
BSpline(l7) = {p54, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85, p86};
p87 = newp;
Point(p87) = {475.0, 500.0, 0.0, 1};
p88 = newp;
Point(p88) = {425.0, 500.0, 0.0, 1};
p89 = newp;
Point(p89) = {375.0, 500.0, 0.0, 1};
p90 = newp;
Point(p90) = {325.0, 500.0, 0.0, 1};
p91 = newp;
Point(p91) = {275.0, 500.00000000000006, 0.0, 1};
p92 = newp;
Point(p92) = {225.0, 500.00000000000006, 0.0, 1};
p93 = newp;
Point(p93) = {175.0, 500.0, 0.0, 1};
p94 = newp;
Point(p94) = {125.00000000000001, 500.0, 0.0, 1};
p95 = newp;
Point(p95) = {75.0, 500.0, 0.0, 1};
p96 = newp;
Point(p96) = {25.0, 500.0, 0.0, 1};
p97 = newp;
Point(p97) = {0.0, 500.0, 0.0, 1};
l8 = newl;
BSpline(l8) = {p86, p87, p88, p89, p90, p91, p92, p93, p94, p95, p96, p97};
p98 = newp;
Point(p98) = {0.0, 490.0, 0.0, 1};
p99 = newp;
Point(p99) = {0.0, 470.0, 0.0, 1};
p100 = newp;
Point(p100) = {0.0, 450.0, 0.0, 1};
p101 = newp;
Point(p101) = {0.0, 430.0, 0.0, 1};
p102 = newp;
Point(p102) = {0.0, 410.0, 0.0, 1};
p103 = newp;
Point(p103) = {0.0, 390.0, 0.0, 1};
p104 = newp;
Point(p104) = {0.0, 370.0, 0.0, 1};
p105 = newp;
Point(p105) = {0.0, 350.0, 0.0, 1};
p106 = newp;
Point(p106) = {0.0, 330.0, 0.0, 1};
p107 = newp;
Point(p107) = {0.0, 310.0, 0.0, 1};
l9 = newl;
BSpline(l9) = {p97, p98, p99, p100, p101, p102, p103, p104, p105, p106, p107, p65};
ll2 = newll;
Line Loop(ll2) = {-l5, l7, l8, l9};
s2 = news;
Plane Surface(s2) = {ll2};
Physical Surface("sandstone") = {s2};
Physical Line("bot_bc") = {l0};
Physical Line("top_bc") = {l8};