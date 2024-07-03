  def write_tensor( self, fname, tensor, band):
    '''
    Write a file with a tensor
    
    Arguments:
        fname (str): Name of the file (written to outputdir)
        tensor (ndarray): Tensor in array form
        band (int): Number of the band that wants to be printed
        
    Returns:
        None
    '''
    attr = self.data_attributes
    if self.rank == 0 :
      from os.path import join
      with open(join(attr['opath'],fname), 'w') as f:
        for i in range(3):
          for j in range(3):
            f.write('%.15e '%(tensor[i,j,band]))
          f.write('\n')
    self.comm.Barrier()