[Mesh]
   file = data/biquadratic.msh
[]

[Variables]
   [./temperature]
   [../]
[]

[GlobalParams]
   variable = temperature
[]

[Kernels]
[./Conduction1]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 10
    block=1
[../]
[]

[BCs]
[./bot]
    type=DirichletBC
    boundary=2
    value=40
[../]
[./top]
    type=DirichletBC
    boundary=3
    value=10
[../]
[]

[Executioner]
   type = Steady
   solve_type = 'Newton'
   l_tol = 1.0e-8
   nl_rel_tol = 1.0e-8
   petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart'
   petsc_options_value = 'hypre    boomeramg      101'
[]

[Outputs]
   exodus = true
   print_perf_log = true
   execute_on = 'timestep_end'
[]

