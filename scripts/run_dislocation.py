
# Continue with the screw-dislocation and NEB workflow.
dislocation_system = BCCScrewDislocation(elem, a0_fit, Cij, calc)

bcc_disl_init = dislocation_system.create_dislocation_object()
# Relax the initial dislocation dipole configuration.
base_system_init, disl_system_init = dislocation_system.relax_dislocation_dipole(
                                            bcc_disl_init, 
                                            disloc_center=[0, 0, 0], 
                                            fmax=0.005, 
                                            optimizer='FIRE'
                                            )
# Convert the relaxed atomman system to ASE for downstream workflows.
dislocation_dipole_ase_initial, properties = disl_system_init.dump('ase_Atoms', return_prop=True)
# Save the differential-displacement map for the initial core location.
dislocation_system.plot_differential_displacement_map(bcc_disl_init, 
                                                      base_system_init, 
                                                      disl_system_init, 
                                                      filename=f"{working_dir}/{elem}_screw_DD_initial.pdf"
                                                      )

# Repeat for the displaced final-state core configuration.
bcc_disl_final = dislocation_system.create_dislocation_object()

base_system_final, disl_system_final = dislocation_system.relax_dislocation_dipole(
                                            bcc_disl_final, 
                                            disloc_center=[a0_fit*np.sqrt(6)/3, 0, 0], 
                                            fmax=0.005, 
                                            optimizer='FIRE'
                                            )
dislocation_dipole_ase_final, properties = disl_system_final.dump('ase_Atoms', return_prop=True)

# Save the differential-displacement map for the final core location.
dislocation_system.plot_differential_displacement_map(bcc_disl_final, 
                                                      base_system_final, 
                                                      disl_system_final, 
                                                      filename=f"{working_dir}/{elem}_screw_DD_final.pdf"
                                                      )

bcc_screw_neb = BCCScrewDislocPeierlsBarrier(dislocation_dipole_ase_initial, 
                                             dislocation_dipole_ase_final, 
                                             calc, model_name='ACE2025', 
                                             Nreplica=11, 
                                             optimizer='FIRE', 
                                             working_dir=working_dir)
bcc_screw_neb.relax_initial_final()
bcc_screw_neb.run_neb(fmax=0.005, spring_constant=0.1)
bcc_screw_neb.plot_barrier(element=f'{elem}', compare_vasp=False)
