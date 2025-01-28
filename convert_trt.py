import time
import torch
from detr.policy import get_n_act_policy

with torch.no_grad():
    TRACING_DEVICE = 'cuda'
    checkpoint_path = "/home/unitree/nw_deploy/parkour/go1_gym_deploy/scripts/ckpts/go1_test/policy_last_alan.pt"
    state_dict = torch.load(checkpoint_path, map_location=TRACING_DEVICE)
    my_model = get_n_act_policy(10)
    ret = my_model.load_state_dict(state_dict)  # Load weights (key may vary)
    print(ret)
    my_model.eval().to(TRACING_DEVICE)  # Set the model to evaluation mode
    ego_view = torch.randn(1, 10, 3, 180, 320).to(TRACING_DEVICE)
    obs_input = torch.randn(1, 753).to(TRACING_DEVICE)
    input_data = (ego_view, obs_input)
    # warming up
    # warm up 
    print("starting warm up")
    for _ in range(10):
        ret = my_model(*input_data)
    print("warm up done")
    print(ret.shape)

    N_iters = 100
    start_cp = time.time()
    for _ in range(N_iters):
        ret = my_model(*input_data)

    end_cp = time.time()
    print(f"Time taken for {N_iters} iterations: {end_cp - start_cp} seconds")
    print(f"Average FPS: {N_iters / (end_cp - start_cp)}")
    # traced_policy = torch.jit.trace(my_model, input_data)
    onnx_path = "/home/unitree/nw_deploy/parkour/go1_gym_deploy/scripts/ckpts/go1_test/test_model.onnx"
    torch.onnx.export(
        my_model,                  # model to export
        input_data,        # inputs of the model,
        onnx_path,        # filename of the ONNX model
        input_names = ['input'],   # the model's input names
        export_params=True,
    )
    from maskclip_onnx.onnx_tensorrt import TensorRTBackend
    trt_engine = TensorRTBackend.prepare(onnx_path,
                                            device='CUDA',
                                            serialize_engine=True,
                                            verbose=False,
                                            serialized_engine_path="/home/unitree/nw_deploy/parkour/go1_gym_deploy/scripts/ckpts/go1_test/test_model.trt")
    output_trt = trt_engine.run(input_data, 'torch_cuda')
    # print("done")
    # print(output_trt)

    print("starting warm up")
    for _ in range(10):
        output_trt = trt_engine.run(input_data, 'torch_cuda')
    print("warm up done")

    N_iters = 100
    start_cp = time.time()
    for _ in range(N_iters):
        output_trt = trt_engine.run(input_data, 'torch_cuda')

    end_cp = time.time()
    print(f"Time taken for {N_iters} iterations: {end_cp - start_cp} seconds")
    print(f"Average FPS: {N_iters / (end_cp - start_cp)}")

    print("Maximum differencce")
    print((output_trt[0] - ret).abs().max())
    import ipdb; ipdb.set_trace()

    pass
