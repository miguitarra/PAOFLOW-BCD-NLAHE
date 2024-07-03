def levi_civita(i, j, k):
  #---------------------
  # Simple function to calculate the levi-civita index
  #---------------------

  if i == j or j == k or k == i:
    return 0
  elif (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    return 1
  else:
    return -1
  
#Naming of files  
axis_names = {0: 'x', 1: 'y', 2: 'z'}

def do_Berry_dipole( data_controller):
  #----------------------
  #
  # Compute Berry curvature dipole:
  # the approach taken is very similar to the way the Berry dipole is calculated. We make use of the a_tensor shapes in order to iterate 
  # through all the possible index. After calculating all the possible values of the Berry curvature, the first moment is taken with 
  # the numpy.gradient() function and specifying the axis depending on what is the variable to derivate over. After this the integral is
  # approximated with a sum.
  # 
  #----------------------
  from .perturb_split import perturb_split
  from .constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL
  from .communication import gather_full
  from .smearing import intgaussian, intmetpax
  from .do_Boltz_tensors import get_tau

  arrays,attributes = data_controller.data_dicts()

  fermi_up,fermi_dw = attributes['fermi_up'],attributes['fermi_dw']
  nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']
  nk = [nk1, nk2, nk3]

  esize = 500
  ene = np.linspace(attributes['eminH'], attributes['emaxH'], esize)

  a_tensor = arrays['a_tensor']
  Om_field = np.empty([3, 3, nk1, nk2, nk3, 2]) #used to be w/ esize too

  a_tensor = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
  for n in range(9):
    #loop through all the possibilities of the "tensor" to create a matrix with three 3d vectors
    ipol = a_tensor[n][0]
    jpol = a_tensor[n][1]

    dks = arrays['dHksp'].shape

    pksp_i = np.zeros((dks[0],dks[2],dks[3],dks[4]),order="C",dtype=complex)
    pksp_j = np.zeros_like(pksp_i)

    for ik in range(dks[0]):
      for ispin in range(dks[4]):
        pksp_i[ik,:,:,ispin],pksp_j[ik,:,:,ispin] = perturb_split(arrays['dHksp'][ik,ipol,:,:,ispin], arrays['dHksp'][ik,jpol,:,:,ispin], arrays['v_k'][ik,:,:,ispin], arrays['degen'][ispin][ik])

    ene,ahc,Om_k = do_Berry_curvature(data_controller, pksp_i, pksp_j)
    
    Om_kps = (np.empty((nk1,nk2,nk3,2), dtype=float) if rank==0 else None)
    
    if rank == 0:
      Om_kps[:,:,:,0] = Om_kps[:,:,:,1] = Om_k[:,:,:]
    #shape of Om_kps is (nk1,nk2,nk3,2)
    Om_field[n//3, n%3] = Om_kps


  dipole = np.empty([3,3,2]) #used to be w/ esize too
  dOm_dk = np.empty([3,3,3,nk1,nk2,nk3,2])
  for r in range(3):
    #iterate through the three vectors
    for c in range(3):
      #iterate through the a1,a2 and a3 component of the vectors        
      #calculation of the gradient, axis points which dimension, k1,k2 or k3 should be derived
      dOm_dk[r,c,0] = np.gradient(Om_field[r,c], axis = 0)
      dOm_dk[r,c,1] = np.gradient(Om_field[r,c], axis = 1)
      dOm_dk[r,c,2] = np.gradient(Om_field[r,c], axis = 2)
  #each term of the dOm_dk array has shape nk1,nk2,nk3,2

  for r in range(3):
    for c in range(3):
      #calculation of the integral by collapsing all axis and just leaving the band values
      dipole[r,c] = np.sum(dOm_dk[r,:,c], axis = (0,1,2,3)) * ( 2*np.pi / nk[c] )**3 

  fdipole1 = 'BCd_bnd1.dat'
  data_controller.write_tensor(fdipole1, dipole, 0)    
  fdipole2 = 'BCd_bnd2.dat'
  data_controller.write_tensor(fdipole2, dipole, 1) 

  tau = 1
  nlac = np.ones((3,3,3,2)) 
  for i in range(3):
    for j in range(3):
      for k in range(3):
        for l in range(2):
          if rank == 0: 
            nlac[i,j,k,l] *=  levi_civita(i,j,k) * dipole[i,j,l] * tau * 1e+4 * ANGSTROM_AU*ELECTRONVOLT_SI**3 / (H_OVER_TPI*attributes['omega'])**2
          else :
            nlac[i,j,k,l] *=  levi_civita(i,j,k) * dipole[i,j,l]
  
  if rank == 0:
    from os.path import join
    fnlaHc = 'nlaHc.dat'
    with open(join(attributes['opath'],fnlaHc), 'w') as f:
      for i in range(3):
        for j in range(3):
          for k in range(3):
            if nlac[i,j,k,0] != 0:
              f.write('%s%s%s %.15e %.15e \n'%(axis_names[i], axis_names[j], axis_names[k], nlac[i,j,k,0], nlac[i,j,k,1]) )
  comm.Barrier()