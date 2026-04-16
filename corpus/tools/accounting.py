from loguru import logger
from rich.console import Console
from rich.table import Table


def get_flops(num_tokens, num_nonembed_params):
    return 6 * num_tokens * num_nonembed_params

def training_statistics(
    num_tokens=10**9, 
    num_nonembed_params=10**9, 
    gpu_type='h100', 
    num_gpus=1,
    mfu=0.4):

    assert 0 < mfu <= 1, 'Invalid model-flops-use, specify MFU between 0 and 1.'

    match gpu_type:
        case 'h100':
            # assume fp8
            flop_second = 3.958 * 10**15
            hourly_cost = 2.29
        case 'b200':
            # assume nvfp4 dense
            flop_second = 9 * 10**15
            hourly_cost = 4.89
        case 'b300':
            # assume nvfp4 dense
            flop_second = 13.5 * 10**15
            hourly_cost = 6.99
        case _:
            raise ValueError(f'Unsupported GPU Type: {gpu_type}')

    flops = get_flops(num_tokens=num_tokens, num_nonembed_params=num_nonembed_params)

    seconds = flops / (flop_second * mfu * num_gpus)
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    dollars = round(hours * hourly_cost * num_gpus)

    model_size = round(num_nonembed_params/10**9, 1) 

    table = Table(title="Training Time Estimate")
    table.add_column("Name", justify="right", style="cyan", no_wrap=True)
    table.add_column("Spec", justify="right", style="green")

    table.add_row(f'Model size', f'{model_size}B params')
    table.add_row(f'Tokens', f'{num_tokens:.2e} tokens')
    table.add_row(f'GPU Type', f'{gpu_type}')
    table.add_row(f'Num GPUs', f'{num_gpus}')
    table.add_row(f'MFU', f'{mfu}')
    table.add_row(f'Days', f'{days:.2e}')
    table.add_row(f'Hours', f'{hours:.2e}')
    table.add_row(f'Minutes', f'{minutes:.2e}')
    table.add_row(f'Seconds', f'{seconds:.2e}')
    table.add_row(f'Cost', f'${dollars}')

    console = Console()
    console.print(table)

def get_token_estimate(gb_size, lang='en', tokenizer_compression_ratio=4, log=True):
    """
    Rough token estimate based on dataset size in gigabytes
    """
    if lang == 'en':
        bytes_per_char = 1 # utf-8
        lanugage = 'English'
    else:
        bytes_per_char = 2 # average utf-8
        lanugage = 'unspecified language'

    num_chars = gb_size * 10**9 / bytes_per_char
    num_tokens = num_chars / tokenizer_compression_ratio
    if log:
        logger.info(f'{gb_size} GB {lanugage} dataset is approximately {num_tokens:.2e} tokens')
    return num_tokens

if __name__ == '__main__':
    data_gb = 100
    num_tokens = get_token_estimate(data_gb)
    training_statistics(
        num_tokens=num_tokens, 
        num_nonembed_params=10 * 10**9, 
        gpu_type='h100', 
        num_gpus=1,
        mfu=0.4)
    

