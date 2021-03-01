[Mesh]
   file = data/fault.msh
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
    thermal_conductivity = 3.1
    block=1
[../]
[./Conduction2]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 3.1
    block=2
[../]
[./Conduction3]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 3.1
    block=3
[../]
[./Conduction4]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 0.9
    block=4
[../]
[./Conduction5]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 3.1
    block=5
[../]
[./Conduction6]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 3
    block=6
[../]
[./Conduction7]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 0.9
    block=7
[../]
[./Conduction8]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 3
    block=8
[../]
[./Conduction9]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 3
    block=9
[../]
[./Conduction10]
    type = DwarfElephantFEThermalConduction
    thermal_conductivity = 3
    block=10
[../]
[]

[BCs]
[./bot]
    type=DirichletBC
    boundary=11
    value=40
[../]
[./top]
    type=DirichletBC
    boundary=12
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

