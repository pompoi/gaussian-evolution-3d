import os
import sys
import torch
import torch.optim as optim
import argparse
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models import DynamicGaussianModel
from src.renderer import GaussianRenderer
from src.training import GaussianTrainer, CombinedLoss
from src.data import TimeEvolvingSceneDataset
from src.utils import load_config, TrainingVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train Dynamic Gaussian Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = TimeEvolvingSceneDataset(
        data_dir=args.data_dir,
        split='train',
        sequence_length=config['training']['sequence_length'],
        sequence_stride=config['training']['sequence_stride'],
        load_observations=config['model']['use_observations']
    )
    
    val_dataset = TimeEvolvingSceneDataset(
        data_dir=args.data_dir,
        split='val',
        sequence_length=config['training']['sequence_length'],
        sequence_stride=config['training']['sequence_stride'],
        load_observations=config['model']['use_observations']
    )
    
    # Initialize model
    model = DynamicGaussianModel(
        num_gaussians=config['model']['num_gaussians'],
        lnn_hidden_size=config['model']['lnn_hidden_size'],
        lnn_num_layers=config['model']['lnn_num_layers'],
        lnn_dropout=config['model']['lnn_dropout'],
        use_observations=config['model']['use_observations'],
        observation_size=config['model']['observation_size'],
        device=device
    )
    
    # Initialize renderer
    renderer = GaussianRenderer(
        image_width=config['renderer']['image_width'],
        image_height=config['renderer']['image_height'],
        background_color=config['renderer']['background_color'],
        tile_size=config['renderer']['tile_size'],
        device=device
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999)
    )
    
    # Initialize loss function
    loss_fn = CombinedLoss(
        l1_weight=config['loss']['l1_weight'],
        ssim_weight=config['loss']['ssim_weight'],
        depth_weight=config['loss']['depth_weight'],
        smoothness_weight=config['loss']['smoothness_weight']
    )
    
    # Initialize visualizer if requested
    visualizer = None
    if args.use_wandb:
        visualizer = TrainingVisualizer(
            use_wandb=True,
            project_name=config['logging']['project_name'],
            save_dir=str(output_dir / 'visualizations')
        )
    
    # Initialize trainer
    trainer = GaussianTrainer(
        model=model,
        renderer=renderer,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        truncation_length=config['training']['truncation_length'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        log_interval=config['logging']['log_interval'],
        patience=config['training']['patience'],
        min_lr=config['training']['min_lr'],
        lr_factor=config['training']['lr_factor'],
        lr_patience=config['training']['lr_patience'],
        lr_threshold=config['training']['lr_threshold'],
        visualizer=visualizer
    )
    
    # Train model
    try:
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            save_path=str(output_dir / 'checkpoints'),
            save_interval=config['logging']['save_interval']
        )
        
        # Save training history
        torch.save(history, output_dir / 'training_history.pt')
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up
        if visualizer is not None:
            visualizer.close()

if __name__ == '__main__':
    main()