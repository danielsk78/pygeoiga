import os
from pygeoiga import myPath
datapath = myPath+"/FE_solvers/data/"

def convert_msh_to_exodus(filepath):
    with open("%s.i"%filepath.split('.')[0], "w") as f:
        f.write("[Mesh]\n"
                "file = %s\n"
                "[]\n"
                "[Outputs]\n"
                "exodus = true\n"
                "[]"%filepath)
    print("DwarfElephant-opt -i %s.i --mesh-only"%filepath.split('.')[0])
    try:
        import subprocess, os
        p1 = os.chdir(datapath)
        #p0 = os.system("conda init")
        #p0 = subprocess.run("source activate moose", shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p1 = subprocess.Popen("DwarfElephant-opt -i %s.i --mesh-only"%filepath.split('.')[0])#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        print("Is moose installed?")
        raise

def create_script(filepath,
                   topology_info:int=None,
                   top_bc:int=None,
                   bot_bc:int=None,
                   left_bc:int=None,
                   right_bc:int = None,
                   geometry: dict = None,
                   kappa = 3, #only if geometry is None
                   ):
    bcs = ""
    #write the boundary conditions
    for bc_id in topology_info.keys():
        if bc_id[-2:] == "bc":
            if bot_bc is not None and bc_id[:3] == "bot":
                line = "[./bot]\n"\
                       "    type=DirichletBC\n" \
                       "    boundary=%s\n"\
                       "    value=%s\n" \
                       "[../]\n"%(topology_info[bc_id], bot_bc)
            elif left_bc is not None and bc_id[:4] == "left":
                line = "[./left]\n" \
                       "    type=DirichletBC\n" \
                       "    boundary=%s\n" \
                       "    value=%s\n" \
                       "[../]\n" % (topology_info[bc_id], left_bc)
            elif top_bc is not None and bc_id[:3] == "top":
                line = "[./top]\n" \
                       "    type=DirichletBC\n" \
                       "    boundary=%s\n" \
                       "    value=%s\n" \
                       "[../]\n" % (topology_info[bc_id], top_bc)
            elif right_bc is not None and bc_id[:5] == "right":
                line = "[./right]\n" \
                       "    type=DirichletBC\n" \
                       "    boundary=%s\n" \
                       "    value=%s\n" \
                       "[../]\n" % (topology_info[bc_id], right_bc)
            else:
                line=""
                print(bc_id + " Not assigned as boundary condition ")
            bcs+=line

    conduction = ""
    if geometry is not None: # run multipatch implementation (Multiple domains)
        for patch_id in geometry.keys():
            kappa = geometry[patch_id].get("kappa")
            line = "[./Conduction%s]\n"\
                   "    type = DwarfElephantFEThermalConduction\n"\
                   "    thermal_conductivity = %s\n"\
                   "    block=%s\n"\
                   "[../]\n" %(topology_info[patch_id], kappa, topology_info[patch_id])
            conduction += line
    else:
        conduction = "[./Conduction1]\n" \
                     "    type = DwarfElephantFEThermalConduction\n" \
                     "    thermal_conductivity = %s\n" \
                     "    block=1\n" \
                     "[../]\n" %kappa

    with open("%s.i" % filepath.split('.')[0], "w") as f:
        # MESH
        f.write("[Mesh]\n"
                "   file = %s\n"
                "[]\n\n"%filepath)
        # VARIABLES AND GLOBAL PARAMS
        f.write("[Variables]\n"
                "   [./temperature]\n"
                "   [../]\n"
                "[]\n\n"
                "[GlobalParams]\n"
                "   variable = temperature\n"
                "[]\n\n")
        # KERNELS
        f.write("[Kernels]\n" +
                conduction +
                "[]\n\n")

        # BOUNDARY CONDITION
        f.write("[BCs]\n"+
                bcs +
                "[]\n\n")

        # EXECUTIONER
        f.write("[Executioner]\n"
                "   type = Steady\n"
                "   solve_type = 'Newton'\n"
                "   l_tol = 1.0e-8\n"
                "   nl_rel_tol = 1.0e-8\n"
                "   petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart'\n"
                "   petsc_options_value = 'hypre    boomeramg      101'\n"
                "[]\n\n")

        # OUTPUTS
        f.write("[Outputs]\n"
                "   exodus = true\n"
                "   print_perf_log = true\n"
                "   execute_on = 'timestep_end'\n"
                "[]\n\n")

    return os.path.abspath("%s.i" % filepath.split('.')[0])


