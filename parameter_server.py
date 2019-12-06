class PS(object):
    def __init__(self, initial_parameter_list):
        # parameter server
        #p_s = [0 for i in range(415310)]
        self.parameter_list = initial_parameter_list

    def set_ps_list(self, ps_list):
        self.parameter_list = ps_list
    def get_ps_list(self):
        return self.parameter_list
    # push weight to participant
    def w_push(self, net_id):
        pass

    # pull weight from participant
    def w_pull(self, net_id):
        pass
