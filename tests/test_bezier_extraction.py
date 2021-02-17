import numpy as np
import matplotlib.pyplot as plt
from pygeoiga.analysis.bezier_extraction import (bivariate_bernstein_basis,
                                                 bezier_extraction_operator,
                                                 bezier_extraction_operator_bivariate)


def test_plot_basis_function():
    from pygeoiga.engine.NURB_engine import basis_function_array_nurbs
    U = np.array([0, 0, 0, 0, 1,1,1, 2, 3, 4, 4, 4, 4])
    resolution = 1000
    degree = len(np.where(U == 0.)[0]) - 1
    N, der = basis_function_array_nurbs(U,
                                                       degree,
                                                       resolution,
                                                       None)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(N)
    fig.show()

def test_univariate_bezier_extraction():
    U = np.array([0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4])
    degree = len(np.where(U == 0.)[0]) - 1

    C = bezier_extraction_operator(U, degree)
    ans = np.array([[[1.        , 0.25      , 0.16666667, 0.16666667],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ]],
       [[0.        , 0.58333333, 0.66666667, 0.58333333],
        [1.        , 0.66666667, 0.66666667, 0.5       ],
        [0.5       , 0.33333333, 0.33333333, 0.        ],
        [0.25      , 0.16666667, 0.16666667, 0.        ]],
       [[0.        , 0.16666667, 0.16666667, 0.25      ],
        [0.        , 0.33333333, 0.33333333, 0.5       ],
        [0.5       , 0.66666667, 0.66666667, 1.        ],
        [0.58333333, 0.66666667, 0.58333333, 0.        ]],
       [[0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.16666667, 0.16666667, 0.25      , 1.        ]]])
    print(C.shape)
    print(C)
    assert np.allclose(ans, C)

def test_univariate_bezier_extraction_2():
    U = np.array([0, 0, 0, 0.1, 0.3, 0.4, 1, 1, 1])
    degree = len(np.where(U == 0.)[0]) - 1

    C = bezier_extraction_operator(U, degree)
    ans = np.array([[[1.        , 0.66666667, 0.33333333, 0.85714286],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ]],
       [[0.        , 0.33333333, 0.66666667, 0.14285714],
        [1.        , 1.        , 1.        , 1.        ],
        [0.66666667, 0.33333333, 0.85714286, 0.        ]],
       [[0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.33333333, 0.66666667, 0.14285714, 1.        ]]])

    assert np.allclose(ans, C)

def test_bezier_extraction_operator_bivariate():
    U = np.array([0, 0, 0, 0.1, 0.3, 0.4, 1, 1, 1])
    degree_u = len(np.where(U == 0.)[0]) - 1

    V = np.array([0, 0, 0, 0.2, 0.65, 0.667, 1, 1, 1])
    degree_v = len(np.where(V == 0.)[0]) - 1

    assert degree_u == degree_v

    C_xi = bezier_extraction_operator(U, degree_u)
    C_eta = bezier_extraction_operator(V, degree_v)

    n_xi = len(C_xi)
    n_eta = len(C_eta)

    C_biv = bezier_extraction_operator_bivariate(C_xi,C_eta, n_xi, n_eta, degree_v)
    print(C_biv)

def test_bezier_extraction_control():
    U = np.array([0, 0, 0,
                 0.0455, 0.0909, 0.1364, 0.1818, 0.2273, 0.2727, 0.3182, 0.3636, 0.4091,
                  0.4545, 0.5000, 0.5455, 0.5909, 0.6364, 0.6818, 0.7273, 0.7727, 0.8182,
                  0.8636, 0.9091, 0.9545, 1, 1, 1])
    degree_u = len(np.where(U == 0.)[0]) - 1

    V = np.array([0, 0, 0, 0.0909, 0.1818, 0.2727, 0.3636, 0.4545, 0.5455, 0.6364, 0.7273,
                  0.8182, 0.9091, 1.0000, 1.0000, 1.0000])

    degree_v = len(np.where(V == 0.)[0]) - 1

    assert degree_u == degree_v

    C_xi = bezier_extraction_operator(U, degree_u)
    C_eta = bezier_extraction_operator(V, degree_v)

    n_xi = len(U)-degree_u-3
    n_eta = len(V)-degree_v-3

    C_biv = bezier_extraction_operator_bivariate(C_xi, C_eta, n_xi, n_eta, degree_v)
    print(C_biv)