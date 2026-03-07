import copy
import os
import shutil
import onnx
import argparse
import subprocess

import torch
import torchaudio
import numpy as np
import onnxruntime as ort
import torch.utils.benchmark as benchmark
from typing import Dict, Iterable, List, Tuple, Union
from copy import deepcopy

from torch_df_streaming import TorchDFPipeline
from typing import Dict, Iterable
from torch.onnx._internal import jit_utils
from loguru import logger
from torch import Tensor
from torch import nn

from df.enhance import parse_epoch_type
from df.deepfilternet3 import ModelParams
from df.enhance import (
    ModelParams,
    df_features,
    enhance,
    get_model_basedir,
    init_df,
    setup_df_argument_parser,
)

from torch_df_streaming_minimal import Encoder, DfDecoder, ErbDecoder

from libdf import DF
torch.manual_seed(0)


def decompose_gru_in_onnx(model_path: str) -> None:
    """Decompose all GRU nodes in an ONNX model into primitive ops in-place.

    Replaces each ONNX GRU node with MatMul / Sigmoid / Tanh / Mul / Add / Sub
    nodes.  hidden_size and input_size are auto-detected from the GRU weight
    tensors so this works for enc, erb_dec, and df_dec regardless of their
    specific hidden dimensions.

    GRU equations (linear_before_reset=1, forward direction):
        zt = sigmoid(Wz*Xt + Rz*Ht-1 + Wbz + Rbz)
        rt = sigmoid(Wr*Xt + Rr*Ht-1 + Wbr + Rbr)
        ht = tanh(Wh*Xt + Wbh + rt*(Rh*Ht-1 + Rbh))
        Ht = (1 - zt) * ht + zt * Ht-1
    """
    from onnx import helper, numpy_helper

    model = onnx.load(model_path)
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    gru_indices = [
        (idx, node)
        for idx, node in enumerate(model.graph.node)
        if node.op_type == "GRU"
    ]

    if not gru_indices:
        logger.info("  No GRU nodes found, skipping decomposition")
        return

    logger.info(f"  Decomposing {len(gru_indices)} GRU node(s) in {model_path}...")

    new_inits: list = []
    replacements: dict = {}

    for gru_counter, (gru_idx, gru_node) in enumerate(gru_indices):
        prefix = f"gru_decomp_{gru_counter}"

        X_name  = gru_node.input[0]
        W_name  = gru_node.input[1]
        R_name  = gru_node.input[2]
        B_name  = gru_node.input[3]
        H0_name = gru_node.input[5]
        Y_name  = gru_node.output[0]
        Yh_name = gru_node.output[1]

        W_full = init_map[W_name][0]  # [3*H, I]
        R_full = init_map[R_name][0]  # [3*H, H]
        B_full = init_map[B_name][0]  # [6*H]

        hidden_size = W_full.shape[0] // 3
        input_size  = W_full.shape[1]

        Wz = W_full[:hidden_size]
        Wr = W_full[hidden_size:2*hidden_size]
        Wh = W_full[2*hidden_size:]
        Rz = R_full[:hidden_size]
        Rr = R_full[hidden_size:2*hidden_size]
        Rh = R_full[2*hidden_size:]
        Wbz = B_full[:hidden_size]
        Wbr = B_full[hidden_size:2*hidden_size]
        Wbh = B_full[2*hidden_size:3*hidden_size]
        Rbz = B_full[3*hidden_size:4*hidden_size]
        Rbr = B_full[4*hidden_size:5*hidden_size]
        Rbh = B_full[5*hidden_size:]

        def add_init(name, data):
            new_inits.append(numpy_helper.from_array(data.astype(np.float32), name=name))
            return name

        gru_const_nodes: list = []

        def add_shape_const(name, int_array):
            tensor = numpy_helper.from_array(np.array(int_array, dtype=np.int64))
            node = helper.make_node(
                "Constant", inputs=[], outputs=[name],
                name=f"{name}_const", value=tensor
            )
            gru_const_nodes.append(node)
            return name

        wz_t = add_init(f"{prefix}/Wz_T", Wz.T)
        wr_t = add_init(f"{prefix}/Wr_T", Wr.T)
        wh_t = add_init(f"{prefix}/Wh_T", Wh.T)
        rz_t = add_init(f"{prefix}/Rz_T", Rz.T)
        rr_t = add_init(f"{prefix}/Rr_T", Rr.T)
        rh_t = add_init(f"{prefix}/Rh_T", Rh.T)

        bz   = add_init(f"{prefix}/bz",   (Wbz + Rbz).reshape(1, hidden_size))
        br   = add_init(f"{prefix}/br",   (Wbr + Rbr).reshape(1, hidden_size))
        wbh  = add_init(f"{prefix}/Wbh",  Wbh.reshape(1, hidden_size))
        rbh  = add_init(f"{prefix}/Rbh",  Rbh.reshape(1, hidden_size))
        ones = add_init(f"{prefix}/ones", np.ones((1, hidden_size), dtype=np.float32))

        h_prev_shape = add_shape_const(f"{prefix}/h_prev_shape", [1, hidden_size])
        x_2d_shape   = add_shape_const(f"{prefix}/x_2d_shape",   [1, input_size])
        yh_shape     = add_shape_const(f"{prefix}/yh_shape",      [1, 1, hidden_size])
        y_shape      = add_shape_const(f"{prefix}/y_shape",       [1, 1, 1, hidden_size])

        p = prefix
        nodes = [
            helper.make_node("Reshape", [H0_name, h_prev_shape], [f"{p}/h_prev"],  name=f"{p}/reshape_h0"),
            helper.make_node("Reshape", [X_name,  x_2d_shape],   [f"{p}/x_2d"],   name=f"{p}/reshape_x"),
            # z gate
            helper.make_node("MatMul",  [f"{p}/x_2d", wz_t],     [f"{p}/xWz"],    name=f"{p}/mm_xWz"),
            helper.make_node("MatMul",  [f"{p}/h_prev", rz_t],   [f"{p}/hRz"],    name=f"{p}/mm_hRz"),
            helper.make_node("Add",     [f"{p}/xWz", f"{p}/hRz"],[f"{p}/z_pre"],  name=f"{p}/add_z_pre"),
            helper.make_node("Add",     [f"{p}/z_pre", bz],       [f"{p}/z_act"],  name=f"{p}/add_z_b"),
            helper.make_node("Sigmoid", [f"{p}/z_act"],           [f"{p}/zt"],     name=f"{p}/sig_z"),
            # r gate
            helper.make_node("MatMul",  [f"{p}/x_2d", wr_t],     [f"{p}/xWr"],    name=f"{p}/mm_xWr"),
            helper.make_node("MatMul",  [f"{p}/h_prev", rr_t],   [f"{p}/hRr"],    name=f"{p}/mm_hRr"),
            helper.make_node("Add",     [f"{p}/xWr", f"{p}/hRr"],[f"{p}/r_pre"],  name=f"{p}/add_r_pre"),
            helper.make_node("Add",     [f"{p}/r_pre", br],       [f"{p}/r_act"],  name=f"{p}/add_r_b"),
            helper.make_node("Sigmoid", [f"{p}/r_act"],           [f"{p}/rt"],     name=f"{p}/sig_r"),
            # h candidate (linear_before_reset=1)
            helper.make_node("MatMul",  [f"{p}/x_2d", wh_t],     [f"{p}/xWh"],    name=f"{p}/mm_xWh"),
            helper.make_node("Add",     [f"{p}/xWh", wbh],        [f"{p}/xWh_b"],  name=f"{p}/add_wbh"),
            helper.make_node("MatMul",  [f"{p}/h_prev", rh_t],   [f"{p}/hRh"],    name=f"{p}/mm_hRh"),
            helper.make_node("Add",     [f"{p}/hRh", rbh],        [f"{p}/hRh_b"],  name=f"{p}/add_rbh"),
            helper.make_node("Mul",     [f"{p}/rt", f"{p}/hRh_b"],[f"{p}/r_gate"], name=f"{p}/mul_r"),
            helper.make_node("Add",     [f"{p}/xWh_b", f"{p}/r_gate"], [f"{p}/h_pre"], name=f"{p}/add_h"),
            helper.make_node("Tanh",    [f"{p}/h_pre"],           [f"{p}/ht"],     name=f"{p}/tanh_h"),
            # Ht = (1-zt)*ht + zt*h_prev
            helper.make_node("Sub",     [ones, f"{p}/zt"],        [f"{p}/1mz"],    name=f"{p}/sub_1mz"),
            helper.make_node("Mul",     [f"{p}/1mz", f"{p}/ht"], [f"{p}/t1"],     name=f"{p}/mul_t1"),
            helper.make_node("Mul",     [f"{p}/zt", f"{p}/h_prev"], [f"{p}/t2"],  name=f"{p}/mul_t2"),
            helper.make_node("Add",     [f"{p}/t1", f"{p}/t2"],   [f"{p}/Ht_2d"], name=f"{p}/add_Ht"),
            # Reshape outputs
            helper.make_node("Reshape", [f"{p}/Ht_2d", yh_shape], [Yh_name],      name=f"{p}/reshape_Yh"),
            helper.make_node("Reshape", [f"{p}/Ht_2d", y_shape],  [Y_name],       name=f"{p}/reshape_Y"),
        ]

        replacements[gru_idx] = (gru_const_nodes, nodes)

    # Rebuild node list in reverse order so insertions don't shift later indices
    nodes_list = list(model.graph.node)
    for gru_idx in sorted(replacements.keys(), reverse=True):
        nodes_list.pop(gru_idx)
        gru_const_nodes, gru_nodes = replacements[gru_idx]
        for i, n in enumerate(gru_const_nodes + gru_nodes):
            nodes_list.insert(gru_idx + i, n)

    del model.graph.node[:]
    model.graph.node.extend(nodes_list)
    model.graph.initializer.extend(new_inits)

    # Remove unused initializers
    used = {inp for node in model.graph.node for inp in node.input}
    used |= {o.name for o in model.graph.output}
    to_remove = [init for init in model.graph.initializer if init.name not in used]
    for init in to_remove:
        model.graph.initializer.remove(init)

    # Remove corresponding graph.input entries
    removed_names = {init.name for init in to_remove}
    inp_to_remove = [inp for inp in model.graph.input if inp.name in removed_names]
    for inp in inp_to_remove:
        model.graph.input.remove(inp)

    del model.graph.value_info[:]

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    logger.info(f"  GRU decomposition complete -> {model_path}")


def validate_decomposed_model(original_path: str, decomposed_path: str,
                              input_dict: Dict[str, Tensor], output_names: List[str]):
    """Compare outputs of original vs decomposed ONNX model."""
    orig_sess = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
    decomp_sess = ort.InferenceSession(decomposed_path, providers=["CPUExecutionProvider"])
    np_inputs = {k: v.numpy() for k, v in input_dict.items()}
    orig_outputs = orig_sess.run(output_names, np_inputs)
    decomp_outputs = decomp_sess.run(output_names, np_inputs)
    all_match = True
    for name, orig, decomp in zip(output_names, orig_outputs, decomp_outputs):
        max_diff = np.max(np.abs(orig - decomp))
        status = "OK" if max_diff < 1e-5 else "MISMATCH"
        logger.info(f"    {name}: max_diff={max_diff:.2e} [{status}]")
        if max_diff >= 1e-5:
            all_match = False
    return all_match

def onnx_simplify(
    path: str, input_data: Dict[str, Tensor], input_shapes: Dict[str, Iterable[int]]
) -> str:
    import onnxsim

    model = onnx.load(path)
    model_simp, check = onnxsim.simplify(
        model,
        input_data=input_data,
        test_input_shapes=input_shapes,
    )
    model_n = os.path.splitext(os.path.basename(path))[0]
    assert check, "Simplified ONNX model could not be validated"
    logger.debug(model_n + ": " + onnx.helper.printable_graph(model.graph))
    try:
        onnx.checker.check_model(model_simp, full_check=True)
    except Exception as e:
        logger.error(f"Failed to simplify model {model_n}. Skipping: {e}")
        return path
    # new_path = os.path.join(os.path.dirname(path), model_n + "_simplified.onnx")
    onnx.save_model(model_simp, path)
    return path

def shapes_dict(
    tensors: Tuple[Tensor], names: Union[Tuple[str], List[str]]
) -> Dict[str, Tuple[int]]:
    if len(tensors) != len(names):
        logger.warning(
            f"  Number of tensors ({len(tensors)}) does not match provided names: {names}"
        )
    return {k: v.shape for (k, v) in zip(names, tensors)}

def onnx_check(path: str, input_dict: Dict[str, Tensor], output_names: Tuple[str]):
    model = onnx.load(path)
    logger.debug(os.path.basename(path) + ": " + onnx.helper.printable_graph(model.graph))
    onnx.checker.check_model(model, full_check=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(output_names, {k: v.numpy() for (k, v) in input_dict.items()})

def export_impl(
    path: str,
    model: torch.nn.Module,
    inputs: Tuple[Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    jit: bool = True,
    opset_version=14,
    check: bool = True,
    simplify: bool = True,
    print_graph: bool = False,
):
    export_dir = os.path.dirname(path)
    if not os.path.isdir(export_dir):
        logger.info(f"Creating export directory: {export_dir}")
        os.makedirs(export_dir)
    model_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Exporting model '{model_name}' to {export_dir}")

    input_shapes = shapes_dict(inputs, input_names)
    logger.info(f"  Input shapes: {input_shapes}")

    outputs = model(*inputs)
    output_shapes = shapes_dict(outputs, output_names)
    logger.info(f"  Output shapes: {output_shapes}")

    if jit:
        model = torch.jit.script(model, example_inputs=[tuple(a for a in inputs)])

    logger.info(f"  Dynamic axis: {dynamic_axes}")
    torch.onnx.export(
        model=deepcopy(model),
        f=path,
        args=inputs,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        output_names=output_names,
        opset_version=opset_version,
        keep_initializers_as_inputs=False,
    )

    input_dict = {k: v for (k, v) in zip(input_names, inputs)}
    if check:
        onnx_outputs = onnx_check(path, input_dict, tuple(output_names))
        for name, out, onnx_out in zip(output_names, outputs, onnx_outputs):
            try:
                np.testing.assert_allclose(
                    out.numpy().squeeze(), onnx_out.squeeze(), rtol=1e-6, atol=1e-5
                )
            except AssertionError as e:
                logger.warning(f"  Elements not close for {name}: {e}")
    if simplify:
        path = onnx_simplify(path, input_dict, shapes_dict(inputs, input_names))
        logger.info(f"  Saved simplified model {path}")
    if print_graph:
        onnx.helper.printable_graph(onnx.load_model(path).graph)

    return outputs

@staticmethod
def remove_conv_block_padding(original_conv: nn.Module) -> nn.Module:
    """
    Remove paddings for convolutions in the original model

    Parameters:
        original_conv:  nn.Module - original convolution module

    Returns:
        output:         nn.Module - new convolution module without paddings
    """
    new_modules = []

    for module in original_conv:
        if not isinstance(module, nn.ConstantPad2d):
            new_modules.append(module)

    return nn.Sequential(*new_modules)

@torch.no_grad()
def main():
    OPSET_VERSION = 14
    model_base_dir="DeepFilterNet3"

    model, state, _ = init_df(
                config_allow_defaults=True,
                model_base_dir=model_base_dir,
                epoch="best",
            )
    model = deepcopy(model).to("cpu")
    model.eval()
    p = ModelParams()

    min_enc = Encoder()
    min_enc.load_state_dict(model.enc.state_dict())
    min_erb_dec = ErbDecoder()
    min_erb_dec.load_state_dict(model.erb_dec.state_dict())
    min_df_dec = DfDecoder()
    min_df_dec.load_state_dict(model.df_dec.state_dict())

    min_enc.erb_conv0 = remove_conv_block_padding(model.enc.erb_conv0)
    min_enc.df_conv0 = remove_conv_block_padding(model.enc.df_conv0)
    min_df_dec.df_convp = remove_conv_block_padding(model.df_dec.df_convp)
    
    min_enc.eval()
    min_erb_dec.eval()
    min_df_dec.eval()

    p = ModelParams()

    # Use streaming-shaped inputs (T = conv_lookahead + 1 = 3 frames)
    time_steps = p.conv_lookahead + 1  # rolling buffer depth
    feat_erb = torch.randn(1, 1, time_steps, p.nb_erb)       # [1, 1, 3, 32]
    feat_spec = torch.randn(1, 2, time_steps, p.nb_df)       # [1, 2, 3, 96]

    # Export encoder
    enc_hidden = torch.randn(1, 1, p.emb_hidden_dim) * 2 - 1  # random hidden state in [-1, 1]
    path = os.path.join("josheen", "enc.onnx")
    inputs = (feat_erb, feat_spec, enc_hidden)
    input_names = ["feat_erb", "feat_spec", "enc_hidden"]
    output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "hidden"]
    e0, e1, e2, e3, emb, c0, lsnr, enc_hidden = export_impl(
        path,
        min_enc,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
        jit=True,
        check=True,
        simplify=True,
        opset_version=OPSET_VERSION,
        print_graph=True,
    )
    # Decompose GRU and validate
    enc_pre_gru = path.replace(".onnx", "_pre_gru.onnx")
    shutil.copy2(path, enc_pre_gru)
    decompose_gru_in_onnx(path)
    enc_input_dict = {k: v for k, v in zip(input_names, inputs)}
    validate_decomposed_model(enc_pre_gru, path, enc_input_dict, output_names)
    np.savez_compressed(
        os.path.join("josheen", "enc_input.npz"),
        feat_erb=feat_erb.numpy(),
        feat_spec=feat_spec.numpy(),
        enc_hidden=enc_hidden.numpy(),
    )
    np.savez_compressed(
        os.path.join("josheen", "enc_output.npz"),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
        emb=emb.numpy(),
        c0=c0.numpy(),
        lsnr=lsnr.numpy(),
        hidden=enc_hidden.numpy(),
    )

    erb_hidden = torch.randn((2, 1, p.emb_hidden_dim)) * 2 - 1  # random hidden state in [-1, 1]

    # Export erb decoder
    np.savez_compressed(
        os.path.join("josheen", "erb_dec_input.npz"),
        emb=emb.numpy(),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
        erb_hidden=erb_hidden.numpy(),
    )
    inputs = (emb.clone(), e3, e2, e1, e0, erb_hidden)
    input_names = ["emb", "e3", "e2", "e1", "e0", "erb_dec_hidden"]
    output_names = ["m", "hidden"]
    path = os.path.join("josheen", "erb_dec.onnx")
    m, erb_hidden_out = export_impl(  # noqa
        path,
        min_erb_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
        jit=True,
        check=True,
        simplify=True,
        opset_version=OPSET_VERSION,
        print_graph=True,
    )
    # Decompose GRU and validate
    erb_pre_gru = path.replace(".onnx", "_pre_gru.onnx")
    shutil.copy2(path, erb_pre_gru)
    decompose_gru_in_onnx(path)
    erb_input_dict = {k: v for k, v in zip(input_names, inputs)}
    validate_decomposed_model(erb_pre_gru, path, erb_input_dict, output_names)
    np.savez_compressed(os.path.join("josheen", "erb_dec_output.npz"), m=m.numpy(), hidden=erb_hidden_out.numpy())

    df_dec_hidden = torch.randn((2, 1, p.emb_hidden_dim)) * 2 - 1  # random hidden state in [-1, 1]
    print("JO-SHEEN emb_hidden_dim {}".format(p.emb_hidden_dim))
    # Export df decoder
    # df_dec expects the rolling c0 buffer with T=df_order frames
    c0_buf = torch.randn(1, p.conv_ch, p.df_order, p.nb_df)  # [1, 64, 5, 96]
    np.savez_compressed(
        os.path.join("josheen", "df_dec_input.npz"), emb=emb.numpy(), c0=c0_buf.numpy(), df_dec_hidden=df_dec_hidden.numpy()
    )
    inputs = (emb.clone(), c0_buf, df_dec_hidden)
    input_names = ["emb", "c0", "df_dec_hidden"]
    output_names = ["coefs", "hidden"]
    path = os.path.join("josheen", "df_dec.onnx")
    coefs, df_hidden_out = export_impl(  # noqa
        path,
        min_df_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
        jit=True,
        check=True,
        simplify=True,
        opset_version=OPSET_VERSION,
        print_graph=True,
    )
    # Decompose GRU and validate
    df_pre_gru = path.replace(".onnx", "_pre_gru.onnx")
    shutil.copy2(path, df_pre_gru)
    decompose_gru_in_onnx(path)
    df_input_dict = {k: v for k, v in zip(input_names, inputs)}
    validate_decomposed_model(df_pre_gru, path, df_input_dict, output_names)
    np.savez_compressed(os.path.join("josheen", "df_dec_output.npz"), coefs=coefs.numpy(), hidden=df_hidden_out.numpy())

if __name__ == "__main__":
    main()
