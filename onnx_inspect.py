import onnx
import numpy as np

def print_model_info(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    
    # Print basic model info
    print(f"\n{'='*50}")
    print(f"Model: {model_path}")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Opset: {[opset.domain+':'+str(opset.version) for opset in model.opset_import]}")
    
    # Print graph inputs
    print("\nInputs:")
    for i, input in enumerate(model.graph.input):
        print(f"  {i}: Name: {input.name}")
        tensor_type = input.type.tensor_type
        if tensor_type.elem_type:
            print(f"     Type: {onnx.TensorProto.DataType.Name(tensor_type.elem_type)}")
        if tensor_type.shape:
            dims = []
            for dim in tensor_type.shape.dim:
                if dim.dim_param:  # Dynamic dimension
                    dims.append(dim.dim_param)
                else:              # Static dimension
                    dims.append(str(dim.dim_value))
            print(f"     Shape: [{', '.join(dims)}]")
    
    # Print graph outputs
    print("\nOutputs:")
    for i, output in enumerate(model.graph.output):
        print(f"  {i}: Name: {output.name}")
        tensor_type = output.type.tensor_type
        if tensor_type.elem_type:
            print(f"     Type: {onnx.TensorProto.DataType.Name(tensor_type.elem_type)}")
        if tensor_type.shape:
            dims = []
            for dim in tensor_type.shape.dim:
                if dim.dim_param:
                    dims.append(dim.dim_param)
                else:
                    dims.append(str(dim.dim_value))
            print(f"     Shape: [{', '.join(dims)}]")
    
    # Print node information (optional)
    print("\nKey Operations:")
    for i, node in enumerate(model.graph.node):
        if i < 5 or i > len(model.graph.node)-5:  # Show first/last 5 ops
            print(f"  {i}: {node.op_type} (Inputs: {node.input}, Outputs: {node.output})")
        elif i == 5:
            print("  ...")
    
    print(f"\nTotal Nodes: {len(model.graph.node)}")
    print(f"Total Initializers (Weights): {len(model.graph.initializer)}")
    print(f"Total Value Infos: {len(model.graph.value_info)}")
    print("="*50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python onnx_inspect.py <model_path.onnx>")
        sys.exit(1)
    
    print_model_info(sys.argv[1])
