// This code was created by pygmsh vunknown.
p0 = newp;
Point(p0) = {0.0, 0.0, 0.0, 95};
p1 = newp;
Point(p1) = {5.0, 0.0, 0.0, 95};
p2 = newp;
Point(p2) = {15.0, 0.0, 0.0, 95};
p3 = newp;
Point(p3) = {25.0, 0.0, 0.0, 95};
p4 = newp;
Point(p4) = {35.0, 0.0, 0.0, 95};
p5 = newp;
Point(p5) = {45.0, 0.0, 0.0, 95};
p6 = newp;
Point(p6) = {55.0, 0.0, 0.0, 95};
p7 = newp;
Point(p7) = {65.0, 0.0, 0.0, 95};
p8 = newp;
Point(p8) = {75.0, 0.0, 0.0, 95};
p9 = newp;
Point(p9) = {85.0, 0.0, 0.0, 95};
p10 = newp;
Point(p10) = {95.0, 0.0, 0.0, 95};
p11 = newp;
Point(p11) = {100.0, 0.0, 0.0, 95};
p12 = newp;
Point(p12) = {107.1, 10.0, 0.0, 95};
p13 = newp;
Point(p13) = {121.30000000000001, 30.0, 0.0, 95};
p14 = newp;
Point(p14) = {135.5, 50.0, 0.0, 95};
p15 = newp;
Point(p15) = {149.7, 70.0, 0.0, 95};
p16 = newp;
Point(p16) = {163.9, 90.0, 0.0, 95};
p17 = newp;
Point(p17) = {178.1, 110.0, 0.0, 95};
p18 = newp;
Point(p18) = {192.3, 130.0, 0.0, 95};
p19 = newp;
Point(p19) = {206.5, 150.0, 0.0, 95};
p20 = newp;
Point(p20) = {220.70000000000002, 170.0, 0.0, 95};
p21 = newp;
Point(p21) = {234.9, 190.0, 0.0, 95};
p22 = newp;
Point(p22) = {242.0, 200.0, 0.0, 95};
p23 = newp;
Point(p23) = {229.9, 200.0, 0.0, 95};
p24 = newp;
Point(p24) = {205.7, 200.0, 0.0, 95};
p25 = newp;
Point(p25) = {181.50000000000003, 200.0, 0.0, 95};
p26 = newp;
Point(p26) = {157.3, 200.0, 0.0, 95};
p27 = newp;
Point(p27) = {133.1, 200.00000000000003, 0.0, 95};
p28 = newp;
Point(p28) = {108.9, 200.00000000000003, 0.0, 95};
p29 = newp;
Point(p29) = {84.70000000000002, 200.0, 0.0, 95};
p30 = newp;
Point(p30) = {60.500000000000014, 200.0, 0.0, 95};
p31 = newp;
Point(p31) = {36.300000000000004, 200.0, 0.0, 95};
p32 = newp;
Point(p32) = {12.100000000000001, 200.0, 0.0, 95};
p33 = newp;
Point(p33) = {0.0, 200.0, 0.0, 95};
p34 = newp;
Point(p34) = {0.0, 190.0, 0.0, 95};
p35 = newp;
Point(p35) = {0.0, 170.0, 0.0, 95};
p36 = newp;
Point(p36) = {0.0, 150.0, 0.0, 95};
p37 = newp;
Point(p37) = {0.0, 130.0, 0.0, 95};
p38 = newp;
Point(p38) = {0.0, 110.0, 0.0, 95};
p39 = newp;
Point(p39) = {0.0, 90.0, 0.0, 95};
p40 = newp;
Point(p40) = {0.0, 70.0, 0.0, 95};
p41 = newp;
Point(p41) = {0.0, 50.0, 0.0, 95};
p42 = newp;
Point(p42) = {0.0, 30.0, 0.0, 95};
p43 = newp;
Point(p43) = {0.0, 10.0, 0.0, 95};
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
Physical Surface("bottom_L") = {s0};
p44 = newp;
Point(p44) = {145.0, 0.0, 0.0, 95};
p45 = newp;
Point(p45) = {235.0, 0.0, 0.0, 95};
p46 = newp;
Point(p46) = {325.0, 0.0, 0.0, 95};
p47 = newp;
Point(p47) = {415.0, 0.0, 0.0, 95};
p48 = newp;
Point(p48) = {505.0, 0.0, 0.0, 95};
p49 = newp;
Point(p49) = {595.0, 0.0, 0.0, 95};
p50 = newp;
Point(p50) = {685.0, 0.0, 0.0, 95};
p51 = newp;
Point(p51) = {775.0, 0.0, 0.0, 95};
p52 = newp;
Point(p52) = {865.0, 0.0, 0.0, 95};
p53 = newp;
Point(p53) = {955.0, 0.0, 0.0, 95};
p54 = newp;
Point(p54) = {1000.0, 0.0, 0.0, 95};
l4 = newl;
BSpline(l4) = {p11, p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54};
p55 = newp;
Point(p55) = {1000.0, 10.0, 0.0, 95};
p56 = newp;
Point(p56) = {1000.0, 30.0, 0.0, 95};
p57 = newp;
Point(p57) = {1000.0, 50.0, 0.0, 95};
p58 = newp;
Point(p58) = {1000.0, 70.0, 0.0, 95};
p59 = newp;
Point(p59) = {1000.0000000000001, 90.0, 0.0, 95};
p60 = newp;
Point(p60) = {1000.0000000000001, 110.0, 0.0, 95};
p61 = newp;
Point(p61) = {1000.0, 130.0, 0.0, 95};
p62 = newp;
Point(p62) = {1000.0, 150.0, 0.0, 95};
p63 = newp;
Point(p63) = {1000.0, 170.0, 0.0, 95};
p64 = newp;
Point(p64) = {1000.0, 190.0, 0.0, 95};
p65 = newp;
Point(p65) = {1000.0, 200.0, 0.0, 95};
l5 = newl;
BSpline(l5) = {p54, p55, p56, p57, p58, p59, p60, p61, p62, p63, p64, p65};
p66 = newp;
Point(p66) = {962.1, 200.0, 0.0, 95};
p67 = newp;
Point(p67) = {886.3, 200.0, 0.0, 95};
p68 = newp;
Point(p68) = {810.5, 200.0, 0.0, 95};
p69 = newp;
Point(p69) = {734.6999999999998, 200.0, 0.0, 95};
p70 = newp;
Point(p70) = {658.8999999999999, 200.00000000000003, 0.0, 95};
p71 = newp;
Point(p71) = {583.0999999999999, 200.00000000000003, 0.0, 95};
p72 = newp;
Point(p72) = {507.29999999999995, 200.0, 0.0, 95};
p73 = newp;
Point(p73) = {431.5, 200.0, 0.0, 95};
p74 = newp;
Point(p74) = {355.70000000000005, 200.0, 0.0, 95};
p75 = newp;
Point(p75) = {279.90000000000003, 200.0, 0.0, 95};
l6 = newl;
BSpline(l6) = {p65, p66, p67, p68, p69, p70, p71, p72, p73, p74, p75, p22};
ll1 = newll;
Line Loop(ll1) = {l4, l5, l6, -l1};
s1 = news;
Plane Surface(s1) = {ll1};
Physical Surface("bottom_R") = {s1};
p76 = newp;
Point(p76) = {245.4, 205.0, 0.0, 95};
p77 = newp;
Point(p77) = {252.20000000000002, 215.0, 0.0, 95};
p78 = newp;
Point(p78) = {259.0, 225.0, 0.0, 95};
p79 = newp;
Point(p79) = {265.8, 235.0, 0.0, 95};
p80 = newp;
Point(p80) = {272.6, 245.0, 0.0, 95};
p81 = newp;
Point(p81) = {279.40000000000003, 255.0, 0.0, 95};
p82 = newp;
Point(p82) = {286.20000000000005, 265.0, 0.0, 95};
p83 = newp;
Point(p83) = {293.0, 275.0, 0.0, 95};
p84 = newp;
Point(p84) = {299.8, 285.0, 0.0, 95};
p85 = newp;
Point(p85) = {306.6, 295.0, 0.0, 95};
p86 = newp;
Point(p86) = {310.0, 300.0, 0.0, 95};
l7 = newl;
BSpline(l7) = {p22, p76, p77, p78, p79, p80, p81, p82, p83, p84, p85, p86};
p87 = newp;
Point(p87) = {294.5, 300.0, 0.0, 95};
p88 = newp;
Point(p88) = {263.5, 300.0, 0.0, 95};
p89 = newp;
Point(p89) = {232.5, 300.0, 0.0, 95};
p90 = newp;
Point(p90) = {201.5, 300.0, 0.0, 95};
p91 = newp;
Point(p91) = {170.5, 300.0, 0.0, 95};
p92 = newp;
Point(p92) = {139.5, 300.0, 0.0, 95};
p93 = newp;
Point(p93) = {108.5, 300.0, 0.0, 95};
p94 = newp;
Point(p94) = {77.5, 300.0, 0.0, 95};
p95 = newp;
Point(p95) = {46.5, 300.0, 0.0, 95};
p96 = newp;
Point(p96) = {15.5, 300.0, 0.0, 95};
p97 = newp;
Point(p97) = {0.0, 300.0, 0.0, 95};
l8 = newl;
BSpline(l8) = {p86, p87, p88, p89, p90, p91, p92, p93, p94, p95, p96, p97};
p98 = newp;
Point(p98) = {0.0, 295.0, 0.0, 95};
p99 = newp;
Point(p99) = {0.0, 285.0, 0.0, 95};
p100 = newp;
Point(p100) = {0.0, 275.0, 0.0, 95};
p101 = newp;
Point(p101) = {0.0, 265.0, 0.0, 95};
p102 = newp;
Point(p102) = {0.0, 255.0, 0.0, 95};
p103 = newp;
Point(p103) = {0.0, 245.0, 0.0, 95};
p104 = newp;
Point(p104) = {0.0, 235.0, 0.0, 95};
p105 = newp;
Point(p105) = {0.0, 225.0, 0.0, 95};
p106 = newp;
Point(p106) = {0.0, 215.0, 0.0, 95};
p107 = newp;
Point(p107) = {0.0, 205.0, 0.0, 95};
l9 = newl;
BSpline(l9) = {p97, p98, p99, p100, p101, p102, p103, p104, p105, p106, p107, p33};
ll2 = newll;
Line Loop(ll2) = {-l2, l7, l8, l9};
s2 = news;
Plane Surface(s2) = {ll2};
Physical Surface("D3") = {s2};
p108 = newp;
Point(p108) = {1000.0, 205.0, 0.0, 95};
p109 = newp;
Point(p109) = {1000.0, 215.0, 0.0, 95};
p110 = newp;
Point(p110) = {1000.0, 225.0, 0.0, 95};
p111 = newp;
Point(p111) = {1000.0, 235.0, 0.0, 95};
p112 = newp;
Point(p112) = {1000.0000000000001, 245.0, 0.0, 95};
p113 = newp;
Point(p113) = {1000.0000000000001, 255.0, 0.0, 95};
p114 = newp;
Point(p114) = {1000.0, 265.0, 0.0, 95};
p115 = newp;
Point(p115) = {1000.0, 275.0, 0.0, 95};
p116 = newp;
Point(p116) = {1000.0, 285.0, 0.0, 95};
p117 = newp;
Point(p117) = {1000.0, 295.0, 0.0, 95};
p118 = newp;
Point(p118) = {1000.0, 300.0, 0.0, 95};
l10 = newl;
BSpline(l10) = {p65, p108, p109, p110, p111, p112, p113, p114, p115, p116, p117, p118};
p119 = newp;
Point(p119) = {965.5, 300.0, 0.0, 95};
p120 = newp;
Point(p120) = {896.5, 300.0, 0.0, 95};
p121 = newp;
Point(p121) = {827.5, 300.0, 0.0, 95};
p122 = newp;
Point(p122) = {758.5, 300.0, 0.0, 95};
p123 = newp;
Point(p123) = {689.5, 300.0, 0.0, 95};
p124 = newp;
Point(p124) = {620.5, 300.0, 0.0, 95};
p125 = newp;
Point(p125) = {551.5, 300.0, 0.0, 95};
p126 = newp;
Point(p126) = {482.5, 300.0, 0.0, 95};
p127 = newp;
Point(p127) = {413.5, 300.0, 0.0, 95};
p128 = newp;
Point(p128) = {344.5, 300.0, 0.0, 95};
l11 = newl;
BSpline(l11) = {p118, p119, p120, p121, p122, p123, p124, p125, p126, p127, p128, p86};
ll3 = newll;
Line Loop(ll3) = {-l6, l10, l11, -l7};
s3 = news;
Plane Surface(s3) = {ll3};
Physical Surface("D4") = {s3};
p129 = newp;
Point(p129) = {324.0, 320.0, 0.0, 95};
p130 = newp;
Point(p130) = {352.0, 360.0, 0.0, 95};
p131 = newp;
Point(p131) = {379.99999999999994, 400.0, 0.0, 95};
p132 = newp;
Point(p132) = {407.99999999999994, 440.0, 0.0, 95};
p133 = newp;
Point(p133) = {436.0, 480.0, 0.0, 95};
p134 = newp;
Point(p134) = {464.0, 520.0, 0.0, 95};
p135 = newp;
Point(p135) = {492.0, 560.0, 0.0, 95};
p136 = newp;
Point(p136) = {520.0, 600.0, 0.0, 95};
p137 = newp;
Point(p137) = {548.0, 640.0, 0.0, 95};
p138 = newp;
Point(p138) = {576.0, 680.0, 0.0, 95};
p139 = newp;
Point(p139) = {590.0, 700.0, 0.0, 95};
l12 = newl;
BSpline(l12) = {p86, p129, p130, p131, p132, p133, p134, p135, p136, p137, p138, p139};
p140 = newp;
Point(p140) = {560.5, 700.0, 0.0, 95};
p141 = newp;
Point(p141) = {501.5, 700.0, 0.0, 95};
p142 = newp;
Point(p142) = {442.5, 700.0, 0.0, 95};
p143 = newp;
Point(p143) = {383.5, 700.0, 0.0, 95};
p144 = newp;
Point(p144) = {324.5, 700.0, 0.0, 95};
p145 = newp;
Point(p145) = {265.5, 700.0, 0.0, 95};
p146 = newp;
Point(p146) = {206.5, 700.0, 0.0, 95};
p147 = newp;
Point(p147) = {147.5, 700.0, 0.0, 95};
p148 = newp;
Point(p148) = {88.5, 700.0, 0.0, 95};
p149 = newp;
Point(p149) = {29.5, 700.0, 0.0, 95};
p150 = newp;
Point(p150) = {0.0, 700.0, 0.0, 95};
l13 = newl;
BSpline(l13) = {p139, p140, p141, p142, p143, p144, p145, p146, p147, p148, p149, p150};
p151 = newp;
Point(p151) = {0.0, 680.0, 0.0, 95};
p152 = newp;
Point(p152) = {0.0, 640.0, 0.0, 95};
p153 = newp;
Point(p153) = {0.0, 600.0, 0.0, 95};
p154 = newp;
Point(p154) = {0.0, 560.0, 0.0, 95};
p155 = newp;
Point(p155) = {0.0, 520.0, 0.0, 95};
p156 = newp;
Point(p156) = {0.0, 480.0, 0.0, 95};
p157 = newp;
Point(p157) = {0.0, 440.0, 0.0, 95};
p158 = newp;
Point(p158) = {0.0, 400.0, 0.0, 95};
p159 = newp;
Point(p159) = {0.0, 360.0, 0.0, 95};
p160 = newp;
Point(p160) = {0.0, 320.0, 0.0, 95};
l14 = newl;
BSpline(l14) = {p150, p151, p152, p153, p154, p155, p156, p157, p158, p159, p160, p97};
ll4 = newll;
Line Loop(ll4) = {-l8, l12, l13, l14};
s4 = news;
Plane Surface(s4) = {ll4};
Physical Surface("D5") = {s4};
p161 = newp;
Point(p161) = {1000.0, 320.0, 0.0, 95};
p162 = newp;
Point(p162) = {1000.0, 360.0, 0.0, 95};
p163 = newp;
Point(p163) = {1000.0, 400.0, 0.0, 95};
p164 = newp;
Point(p164) = {1000.0, 440.0, 0.0, 95};
p165 = newp;
Point(p165) = {1000.0000000000001, 480.0, 0.0, 95};
p166 = newp;
Point(p166) = {1000.0000000000001, 520.0, 0.0, 95};
p167 = newp;
Point(p167) = {1000.0, 560.0, 0.0, 95};
p168 = newp;
Point(p168) = {1000.0, 600.0, 0.0, 95};
p169 = newp;
Point(p169) = {1000.0, 640.0, 0.0, 95};
p170 = newp;
Point(p170) = {1000.0, 680.0, 0.0, 95};
p171 = newp;
Point(p171) = {1000.0, 700.0, 0.0, 95};
l15 = newl;
BSpline(l15) = {p118, p161, p162, p163, p164, p165, p166, p167, p168, p169, p170, p171};
p172 = newp;
Point(p172) = {979.5, 700.0, 0.0, 95};
p173 = newp;
Point(p173) = {938.5, 700.0, 0.0, 95};
p174 = newp;
Point(p174) = {897.5, 700.0, 0.0, 95};
p175 = newp;
Point(p175) = {856.5, 700.0, 0.0, 95};
p176 = newp;
Point(p176) = {815.5, 700.0, 0.0, 95};
p177 = newp;
Point(p177) = {774.5, 700.0, 0.0, 95};
p178 = newp;
Point(p178) = {733.5, 700.0, 0.0, 95};
p179 = newp;
Point(p179) = {692.5, 700.0, 0.0, 95};
p180 = newp;
Point(p180) = {651.5, 700.0, 0.0, 95};
p181 = newp;
Point(p181) = {610.5, 700.0, 0.0, 95};
l16 = newl;
BSpline(l16) = {p171, p172, p173, p174, p175, p176, p177, p178, p179, p180, p181, p139};
ll5 = newll;
Line Loop(ll5) = {-l11, l15, l16, -l12};
s5 = news;
Plane Surface(s5) = {ll5};
Physical Surface("D6") = {s5};
p182 = newp;
Point(p182) = {593.5, 705.0, 0.0, 95};
p183 = newp;
Point(p183) = {600.5, 715.0, 0.0, 95};
p184 = newp;
Point(p184) = {607.5, 725.0, 0.0, 95};
p185 = newp;
Point(p185) = {614.5, 735.0, 0.0, 95};
p186 = newp;
Point(p186) = {621.5, 745.0, 0.0, 95};
p187 = newp;
Point(p187) = {628.5, 755.0, 0.0, 95};
p188 = newp;
Point(p188) = {635.5, 765.0, 0.0, 95};
p189 = newp;
Point(p189) = {642.5, 775.0, 0.0, 95};
p190 = newp;
Point(p190) = {649.5, 785.0, 0.0, 95};
p191 = newp;
Point(p191) = {656.5, 795.0, 0.0, 95};
p192 = newp;
Point(p192) = {660.0, 800.0, 0.0, 95};
l17 = newl;
BSpline(l17) = {p139, p182, p183, p184, p185, p186, p187, p188, p189, p190, p191, p192};
p193 = newp;
Point(p193) = {627.0, 800.0, 0.0, 95};
p194 = newp;
Point(p194) = {561.0, 800.0, 0.0, 95};
p195 = newp;
Point(p195) = {495.0, 800.0, 0.0, 95};
p196 = newp;
Point(p196) = {429.0, 800.0, 0.0, 95};
p197 = newp;
Point(p197) = {363.0, 800.0000000000001, 0.0, 95};
p198 = newp;
Point(p198) = {297.0, 800.0000000000001, 0.0, 95};
p199 = newp;
Point(p199) = {231.0, 800.0, 0.0, 95};
p200 = newp;
Point(p200) = {165.0, 800.0, 0.0, 95};
p201 = newp;
Point(p201) = {99.00000000000001, 800.0, 0.0, 95};
p202 = newp;
Point(p202) = {33.0, 800.0, 0.0, 95};
p203 = newp;
Point(p203) = {0.0, 800.0, 0.0, 95};
l18 = newl;
BSpline(l18) = {p192, p193, p194, p195, p196, p197, p198, p199, p200, p201, p202, p203};
p204 = newp;
Point(p204) = {0.0, 795.0, 0.0, 95};
p205 = newp;
Point(p205) = {0.0, 785.0, 0.0, 95};
p206 = newp;
Point(p206) = {0.0, 775.0, 0.0, 95};
p207 = newp;
Point(p207) = {0.0, 765.0, 0.0, 95};
p208 = newp;
Point(p208) = {0.0, 755.0, 0.0, 95};
p209 = newp;
Point(p209) = {0.0, 745.0, 0.0, 95};
p210 = newp;
Point(p210) = {0.0, 735.0, 0.0, 95};
p211 = newp;
Point(p211) = {0.0, 725.0, 0.0, 95};
p212 = newp;
Point(p212) = {0.0, 715.0, 0.0, 95};
p213 = newp;
Point(p213) = {0.0, 705.0, 0.0, 95};
l19 = newl;
BSpline(l19) = {p203, p204, p205, p206, p207, p208, p209, p210, p211, p212, p213, p150};
ll6 = newll;
Line Loop(ll6) = {-l13, l17, l18, l19};
s6 = news;
Plane Surface(s6) = {ll6};
Physical Surface("D7") = {s6};
p214 = newp;
Point(p214) = {1000.0, 705.0, 0.0, 95};
p215 = newp;
Point(p215) = {1000.0, 715.0, 0.0, 95};
p216 = newp;
Point(p216) = {1000.0, 725.0, 0.0, 95};
p217 = newp;
Point(p217) = {1000.0, 735.0, 0.0, 95};
p218 = newp;
Point(p218) = {1000.0000000000001, 745.0, 0.0, 95};
p219 = newp;
Point(p219) = {1000.0000000000001, 755.0, 0.0, 95};
p220 = newp;
Point(p220) = {1000.0, 765.0, 0.0, 95};
p221 = newp;
Point(p221) = {1000.0, 775.0, 0.0, 95};
p222 = newp;
Point(p222) = {1000.0, 785.0, 0.0, 95};
p223 = newp;
Point(p223) = {1000.0, 795.0, 0.0, 95};
p224 = newp;
Point(p224) = {1000.0, 800.0, 0.0, 95};
l20 = newl;
BSpline(l20) = {p171, p214, p215, p216, p217, p218, p219, p220, p221, p222, p223, p224};
p225 = newp;
Point(p225) = {983.0, 800.0, 0.0, 95};
p226 = newp;
Point(p226) = {949.0, 800.0, 0.0, 95};
p227 = newp;
Point(p227) = {915.0, 800.0, 0.0, 95};
p228 = newp;
Point(p228) = {881.0, 800.0, 0.0, 95};
p229 = newp;
Point(p229) = {847.0, 800.0000000000001, 0.0, 95};
p230 = newp;
Point(p230) = {813.0, 800.0000000000001, 0.0, 95};
p231 = newp;
Point(p231) = {778.9999999999999, 800.0, 0.0, 95};
p232 = newp;
Point(p232) = {744.9999999999999, 800.0, 0.0, 95};
p233 = newp;
Point(p233) = {711.0, 800.0, 0.0, 95};
p234 = newp;
Point(p234) = {677.0, 800.0, 0.0, 95};
l21 = newl;
BSpline(l21) = {p224, p225, p226, p227, p228, p229, p230, p231, p232, p233, p234, p192};
ll7 = newll;
Line Loop(ll7) = {-l16, l20, l21, -l17};
s7 = news;
Plane Surface(s7) = {ll7};
Physical Surface("D8") = {s7};
p235 = newp;
Point(p235) = {667.0, 810.0, 0.0, 95};
p236 = newp;
Point(p236) = {681.0, 830.0, 0.0, 95};
p237 = newp;
Point(p237) = {695.0, 849.9999999999999, 0.0, 95};
p238 = newp;
Point(p238) = {709.0, 869.9999999999999, 0.0, 95};
p239 = newp;
Point(p239) = {723.0, 890.0, 0.0, 95};
p240 = newp;
Point(p240) = {737.0, 910.0, 0.0, 95};
p241 = newp;
Point(p241) = {751.0, 930.0, 0.0, 95};
p242 = newp;
Point(p242) = {765.0, 950.0, 0.0, 95};
p243 = newp;
Point(p243) = {779.0, 970.0, 0.0, 95};
p244 = newp;
Point(p244) = {793.0, 990.0, 0.0, 95};
p245 = newp;
Point(p245) = {800.0, 1000.0, 0.0, 95};
l22 = newl;
BSpline(l22) = {p192, p235, p236, p237, p238, p239, p240, p241, p242, p243, p244, p245};
p246 = newp;
Point(p246) = {760.0, 1000.0, 0.0, 95};
p247 = newp;
Point(p247) = {680.0, 1000.0, 0.0, 95};
p248 = newp;
Point(p248) = {600.0, 1000.0, 0.0, 95};
p249 = newp;
Point(p249) = {520.0, 1000.0, 0.0, 95};
p250 = newp;
Point(p250) = {440.0, 1000.0000000000001, 0.0, 95};
p251 = newp;
Point(p251) = {360.0, 1000.0000000000001, 0.0, 95};
p252 = newp;
Point(p252) = {280.0, 1000.0, 0.0, 95};
p253 = newp;
Point(p253) = {200.0, 1000.0, 0.0, 95};
p254 = newp;
Point(p254) = {120.0, 1000.0, 0.0, 95};
p255 = newp;
Point(p255) = {40.0, 1000.0, 0.0, 95};
p256 = newp;
Point(p256) = {0.0, 1000.0, 0.0, 95};
l23 = newl;
BSpline(l23) = {p245, p246, p247, p248, p249, p250, p251, p252, p253, p254, p255, p256};
p257 = newp;
Point(p257) = {0.0, 990.0, 0.0, 95};
p258 = newp;
Point(p258) = {0.0, 970.0, 0.0, 95};
p259 = newp;
Point(p259) = {0.0, 950.0, 0.0, 95};
p260 = newp;
Point(p260) = {0.0, 930.0, 0.0, 95};
p261 = newp;
Point(p261) = {0.0, 910.0, 0.0, 95};
p262 = newp;
Point(p262) = {0.0, 890.0, 0.0, 95};
p263 = newp;
Point(p263) = {0.0, 869.9999999999999, 0.0, 95};
p264 = newp;
Point(p264) = {0.0, 849.9999999999999, 0.0, 95};
p265 = newp;
Point(p265) = {0.0, 830.0, 0.0, 95};
p266 = newp;
Point(p266) = {0.0, 810.0, 0.0, 95};
l24 = newl;
BSpline(l24) = {p256, p257, p258, p259, p260, p261, p262, p263, p264, p265, p266, p203};
ll8 = newll;
Line Loop(ll8) = {-l18, l22, l23, l24};
s8 = news;
Plane Surface(s8) = {ll8};
Physical Surface("top_L") = {s8};
p267 = newp;
Point(p267) = {1000.0, 810.0, 0.0, 95};
p268 = newp;
Point(p268) = {1000.0, 830.0, 0.0, 95};
p269 = newp;
Point(p269) = {1000.0, 849.9999999999999, 0.0, 95};
p270 = newp;
Point(p270) = {1000.0, 869.9999999999999, 0.0, 95};
p271 = newp;
Point(p271) = {1000.0000000000001, 890.0, 0.0, 95};
p272 = newp;
Point(p272) = {1000.0000000000001, 910.0, 0.0, 95};
p273 = newp;
Point(p273) = {1000.0, 930.0, 0.0, 95};
p274 = newp;
Point(p274) = {1000.0, 950.0, 0.0, 95};
p275 = newp;
Point(p275) = {1000.0, 970.0, 0.0, 95};
p276 = newp;
Point(p276) = {1000.0, 990.0, 0.0, 95};
p277 = newp;
Point(p277) = {1000.0, 1000.0, 0.0, 95};
l25 = newl;
BSpline(l25) = {p224, p267, p268, p269, p270, p271, p272, p273, p274, p275, p276, p277};
p278 = newp;
Point(p278) = {990.0, 1000.0, 0.0, 95};
p279 = newp;
Point(p279) = {970.0, 1000.0, 0.0, 95};
p280 = newp;
Point(p280) = {950.0, 1000.0, 0.0, 95};
p281 = newp;
Point(p281) = {930.0, 1000.0, 0.0, 95};
p282 = newp;
Point(p282) = {910.0, 1000.0000000000001, 0.0, 95};
p283 = newp;
Point(p283) = {890.0, 1000.0000000000001, 0.0, 95};
p284 = newp;
Point(p284) = {869.9999999999999, 1000.0, 0.0, 95};
p285 = newp;
Point(p285) = {849.9999999999999, 1000.0, 0.0, 95};
p286 = newp;
Point(p286) = {830.0, 1000.0, 0.0, 95};
p287 = newp;
Point(p287) = {810.0, 1000.0, 0.0, 95};
l26 = newl;
BSpline(l26) = {p277, p278, p279, p280, p281, p282, p283, p284, p285, p286, p287, p245};
ll9 = newll;
Line Loop(ll9) = {-l21, l25, l26, -l22};
s9 = news;
Plane Surface(s9) = {ll9};
Physical Surface("top_R") = {s9};
Physical Line("bot_bc") = {l0, l4};
Physical Line("top_bc") = {l23, l26};