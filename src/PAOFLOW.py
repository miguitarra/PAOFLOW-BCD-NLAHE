  def nonlinear_Hall ( self, emin=-1., emax=1., fermi_up=1., fermi_dw=-1.):
    '''
    Calculate the nonlinear anomalous Hall conductivity

    Arguments:
        emin (float): The minimum energy in the range
        emax (float): The maximum energy in the range
        fermi_up (float): The upper limit of the occupied energy range
        fermi_dw (float): The lower limit of the occupied energy range

    Returns:
        None
    '''
    from .defs.do_Hall_modified import do_Berry_dipole

    arrays,attr = self.data_controller.data_dicts()

    attr['eminH'] = emin
    attr['emaxH'] = emax

    if 'fermi_up' not in attr: attr['fermi_up'] = fermi_up
    if 'fermi_dw' not in attr: attr['fermi_dw'] = fermi_dw

    do_Berry_dipole(self.data_controller)

    self.report_module_time('Nonlinear Anomalous Hall Conductivity')
