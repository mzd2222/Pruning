import torch
import time
from tqdm import tqdm
# import models

def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency

def test_latency(model_name, model):
    input_size = (1, 3, 224, 224)
    latency = compute_latency_ms_pytorch(model, input_size, iterations=100, device='cuda')
    string = 'model:{}, | latency: {}'.format(model_name, latency)
    print(string)
