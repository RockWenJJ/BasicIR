import argparse
import importlib
import torch
import time
from thop import profile
from basicir.utils.options import parse

def get_model_complexity_info(model, input_size=(3, 256, 256), n_runs=100):
    """Calculate parameters, FLOPs and inference time for a given model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(1, *input_size).to(device)
    model = model.to(device)
    
    # Calculate MACs and params
    macs, params = profile(model, inputs=(input_tensor,))
    
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
    
    # Convert to readable format
    def human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    return {
        'params': human_format(params),
        'macs': human_format(macs),
        'params_raw': params,
        'macs_raw': macs,
        'inference_time': avg_time  # in milliseconds
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, help='Architecture name')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--input_size', type=int, default=256, help='Input size (assuming square input)')
    parser.add_argument('--n_runs', type=int, default=100, help='Number of runs for averaging inference time')
    args = parser.parse_args()

    try:
        if args.config:
            # Load config file
            opt = parse(args.config, is_train=False)
            
            # Get network_g parameters from config
            if 'network_g' not in opt:
                raise ValueError("Config file must contain 'network_g' settings")
            
            network_opt = opt['network_g']
            network_type = network_opt.pop('type')
            if network_type != args.arch:
                print(f"Warning: Config file specifies architecture '{network_type}' "
                      f"but requested architecture is '{args.arch}'")

            # Dynamically import the architecture
            arch_module = importlib.import_module('basicir.models.archs')
            model_class = getattr(arch_module, args.arch)
            
            # Initialize model with parameters from config
            model = model_class(**network_opt)
        else:
            # If no config is provided, initialize the default architecture
            print("No config provided. Initializing default architecture.")
            arch_module = importlib.import_module('basicir.models.archs')
            model_class = getattr(arch_module, args.arch)
            model = model_class()

        # Get complexity info
        complexity_info = get_model_complexity_info(
            model, 
            input_size=(3, args.input_size, args.input_size),
            n_runs=args.n_runs
        )
        
        print('=' * 50)
        print(f'Architecture: {args.arch}')
        print(f'Config file: {args.config if args.config else "None (default initialization)"}')
        print(f'Input size: ({3} x {args.input_size} x {args.input_size})')
        print('-' * 50)
        print(f'Parameters: {complexity_info["params"]}')
        print(f'MACs: {complexity_info["macs"]}')
        print(f'Average inference time: {complexity_info["inference_time"]:.2f} ms')
        print('=' * 50)
        
    except Exception as e:
        print(f'Error: {str(e)}')
        print(f'Failed to analyze architecture: {args.arch}')
        return

if __name__ == '__main__':
    main() 