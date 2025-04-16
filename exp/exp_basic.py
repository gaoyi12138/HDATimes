from models import AutoTimes_Llama, AutoTimes_Gpt2, AutoTimes_Gpt2_HD1, AutoTimes_Gpt2_HD2, AutoTimes_Opt_1b, \
    AutoTimes_Gpt2_dct, AutoTimes_Gpt2_HD2_dct, ns_AutoTimes_Gpt2, ns_AutoTimes_Gpt2_HD2, AutoTimes_Gpt2_PAM, \
    AutoTimes_Gpt2_danet, AutoTimes_danet_Gpt2, AutoTimes_Gpt2_HD2_danet, AutoTimes_psda_Gpt2


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'AutoTimes_Llama': AutoTimes_Llama,
            'AutoTimes_Gpt2': AutoTimes_Gpt2,
            'AutoTimes_Opt_1b': AutoTimes_Opt_1b,
            'AutoTimes_Gpt2_HD1':AutoTimes_Gpt2_HD1,
            'AutoTimes_Gpt2_HD2':AutoTimes_Gpt2_HD2,
            'AutoTimes_Gpt2_dct':AutoTimes_Gpt2_dct,
            'AutoTimes_Gpt2_HD2_dct':AutoTimes_Gpt2_HD2_dct,
            'ns_AutoTimes_Gpt2': ns_AutoTimes_Gpt2,
            'ns_AutoTimes_Gpt2_HD2': ns_AutoTimes_Gpt2_HD2,
            'AutoTimes_Gpt2_PAM': AutoTimes_Gpt2_PAM,
            'AutoTimes_Gpt2_danet': AutoTimes_Gpt2_danet,
            'AutoTimes_danet_Gpt2':AutoTimes_danet_Gpt2,
            'AutoTimes_Gpt2_HD2_danet':AutoTimes_Gpt2_HD2_danet,
            'AutoTimes_psda_Gpt2':AutoTimes_psda_Gpt2
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
