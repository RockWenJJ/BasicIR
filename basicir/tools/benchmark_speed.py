import argparse
import importlib
import torch
import time
from basicir.utils.options import parse

def benchmark_inference_time(model, input_size=(3, 256, 256), n_runs=100):
    """Calculate inference time for a given model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(1, *input_size).to(device)
    model = model.to(device)
    
    # Measure inference time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        # Measure time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(n_runs):
                _ = model(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
        else:
            start_time = time.time()
            for _ in range(n_runs):
                _ = model(input_tensor)
            end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs * 1000  # Convert to milliseconds
    return avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, help='Architecture name')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--input_size', type=int, default=256, help='Input size (assuming square input)')
    parser.add_argument('--n_runs', type=int, default=100, help='Number of runs for averaging inference time')
    args = parser.parse_args()

    try:
        if args.config:
            opt = parse(args.config, is_train=False)
            if 'network_g' not in opt:
                raise ValueError("Config file must contain 'network_g' settings")
            
            network_opt = opt['network_g']
            network_type = network_opt.pop('type')
            if network_type != args.arch:
                print(f"Warning: Config file specifies architecture '{network_type}' "
                      f"but requested architecture is '{args.arch}'")

            arch_module = importlib.import_module('basicir.models.archs')
            model_class = getattr(arch_module, args.arch)
            model = model_class(**network_opt)
        else:
            print("No config provided. Initializing default architecture.")
            arch_module = importlib.import_module('basicir.models.archs')
            model_class = getattr(arch_module, args.arch)
            model = model_class()

        # Get inference time
        avg_time = benchmark_inference_time(
            model, 
            input_size=(3, args.input_size, args.input_size),
            n_runs=args.n_runs
        )
        
        print('=' * 50)
        print(f'Architecture: {args.arch}')
        print(f'Config file: {args.config if args.config else "None (default initialization)"}')
        print(f'Input size: ({3} x {args.input_size} x {args.input_size})')
        print(f'Number of runs: {args.n_runs}')
        print('-' * 50)
        print(f'Average inference time: {avg_time:.2f} ms')
        print('=' * 50)
        
    except Exception as e:
        print(f'Error: {str(e)}')
        print(f'Failed to benchmark architecture: {args.arch}')
        return

if __name__ == '__main__':
    main() 