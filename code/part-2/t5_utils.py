import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)
    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best, optimizer=None, scheduler=None, epoch=None, best_loss=None):
    mkdir(checkpoint_dir)
    filename = 'best_model.pt' if best else 'latest_model.pt'
    path = os.path.join(checkpoint_dir, filename)
    state = {'model_state_dict': model.state_dict()}
    if not best and optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        state['epoch'] = epoch
        state['best_loss'] = best_loss
    torch.save(state, path)

def load_model_from_checkpoint(args, best):
    model = initialize_model(args)
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    filename = 'best_model.pt' if best else 'latest_model.pt'
    path = os.path.join(checkpoint_dir, filename)
    state = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    return model

def resume_from_checkpoint(args, model, optimizer, scheduler):
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    path = os.path.join(checkpoint_dir, 'latest_model.pt')
    if not os.path.exists(path):
        return 0, float('inf')
    state = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' not in state:
        return 0, float('inf')
    model.load_state_dict(state['model_state_dict'])
    if 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    start_epoch = state.get('epoch', -1) + 1
    best_loss = state.get('best_loss', float('inf'))
    print(f"Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.4f}")
    return start_epoch, best_loss

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

