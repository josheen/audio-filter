import torch


class ModelParams(DfParams):
    section = "deepfilternet"

    def __init__(self):
        super().__init__()
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.conv_depthwise: bool = config(
            "CONV_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.convt_depthwise: bool = config(
            "CONVT_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.conv_kernel: List[int] = config(
            "CONV_KERNEL", cast=Csv(int), default=(1, 3), section=self.section  # type: ignore
        )
        self.conv_kernel_inp: List[int] = config(
            "CONV_KERNEL_INP", cast=Csv(int), default=(3, 3), section=self.section  # type: ignore
        )
        self.emb_hidden_dim: int = config(
            "EMB_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.emb_num_layers: int = config(
            "EMB_NUM_LAYERS", cast=int, default=2, section=self.section
        )
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_gru_skip: str = config("DF_GRU_SKIP", default="none", section=self.section)
        self.df_output_layer: str = config(
            "DF_OUTPUT_LAYER", default="linear", section=self.section
        )
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section
        )
        self.enc_concat: bool = config("ENC_CONCAT", cast=bool, default=False, section=self.section)
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=self.section)
        self.df_n_iter: int = config("DF_N_ITER", cast=int, default=2, section=self.section)
        self.gru_type: str = config("GRU_TYPE", default="grouped", section=self.section)
        self.gru_groups: int = config("GRU_GROUPS", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.group_shuffle: bool = config(
            "GROUP_SHUFFLE", cast=bool, default=True, section=self.section
        )
        self.dfop_method: str = config(
            "DFOP_METHOD", cast=str, default="real_unfold", section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)
model = torch.load('dfn2.bin')


