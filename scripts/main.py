import argparse
import ast
import torch
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from scripts.grid_search.grid_searcher import GridSearcher
from scripts.baseline.baseline_tester import BaselineTester
from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer
from model_utils.models.siamese import SiameseCLIPModelPairs, SiameseCLIPTriplet
from model_utils.models.supcon import SiameseCLIPSupCon
from model_utils.models.infonce import SiameseCLIPInfoNCE

def main():
    parser = argparse.ArgumentParser(description='CLIP-based text similarity training and evaluation')
    parser.add_argument('--mode', type=str, 
                      choices=['train', 'grid_search', 'bayesian', 'random', 'optuna', 'pbt', 'compare', 'baseline'], 
                      required=True,
                      help='Mode to run: train, grid_search, bayesian, random, optuna, pbt, compare, or baseline')
    parser.add_argument('--reference_filepath', type=str, required=True,
                      help='Path to reference data')
    parser.add_argument('--test_reference_filepath', type=str, required=True,
                      help='Path to test reference data')
    parser.add_argument('--test_filepath', type=str, required=True,
                      help='Path to test data')
    parser.add_argument('--model_type', type=str, choices=['pair', 'triplet', 'supcon', 'infonce'], default='pair',
                      help='Model type: pair, triplet, supcon, or infonce')
    parser.add_argument('--loss_type', type=str, choices=['cosine', 'euclidean', 'hybrid', 'supcon', 'infonce'], default='cosine',
                      help='Loss function type')
    parser.add_argument('--medium_filepath', type=str,
                      help='Path to medium data (optional)')
    parser.add_argument('--easy_filepath', type=str,
                      help='easy to medium data (optional)')
    
    # Grid search parameters
    parser.add_argument('--lrs', type=str, default='[1e-4]',
                      help='Learning rates to try (for grid search)')
    parser.add_argument('--batch_sizes', type=str, default='[32]',
                      help='Batch sizes to try (for grid search)')
    parser.add_argument('--margins', type=str, default='[0.5]',
                      help='Margins to try (for grid search)')
    parser.add_argument('--internal_layer_sizes', type=str, default='[128]',
                      help='Internal layer sizes to try (for grid search)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--curriculum', type=str, default=None,
                      help='Curriculum learning mode')
    parser.add_argument('--log_dir', type=str, default='/content/drive/MyDrive/Project_2_Business_Names/Summer 2025/code',
                      help='Directory to save results')
    parser.add_argument('--temperature', type=float, default=0.07,
                      help='Temperature parameter for SupCon/InfoNCE loss (default: 0.07)')
    
    # Hyperparameter optimization parameters
    parser.add_argument('--n_trials', type=int, default=50,
                      help='Number of trials for optimization methods')
    parser.add_argument('--n_calls', type=int, default=50,
                      help='Number of calls for Bayesian optimization')
    parser.add_argument('--n_random_starts', type=int, default=10,
                      help='Number of random starts for Bayesian optimization')
    parser.add_argument('--population_size', type=int, default=8,
                      help='Population size for PBT')
    parser.add_argument('--generations', type=int, default=10,
                      help='Number of generations for PBT')
    parser.add_argument('--epochs_per_generation', type=int, default=5,
                      help='Epochs per generation for PBT')
    parser.add_argument('--evolution_frequency', type=int, default=2,
                      help='Evolution frequency for PBT')
    parser.add_argument('--sampler', type=str, choices=['tpe', 'random', 'cmaes'], default='tpe',
                      help='Sampler for Optuna optimization')
    parser.add_argument('--pruner', type=str, choices=['median', 'hyperband', 'none'], default='median',
                      help='Pruner for Optuna optimization')
    parser.add_argument('--study_name', type=str,
                      help='Study name for Optuna optimization')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'baseline':
        # Test raw CLIP performance
        tester = BaselineTester()
        results_df, metrics = tester.test(args.test_reference_filepath, args.test_filepath)
        print("\nBaseline Results:")
        print(metrics)

    elif args.mode == 'train':
        # Single training run
        if args.model_type == 'pair':
            model_class = SiameseCLIPModelPairs
        elif args.model_type == 'triplet':
            model_class = SiameseCLIPTriplet
        elif args.model_type == 'supcon':
            model_class = SiameseCLIPSupCon
        else:  # infonce
            model_class = SiameseCLIPInfoNCE
            
        model = model_class(embedding_dim=512, projection_dim=128).to(device)
        
        # Get appropriate loss class
        if args.model_type == 'pair':
            if args.loss_type == 'cosine':
                from model_utils.loss.pair_losses import CosineLoss
                criterion = CosineLoss(margin=0.5)
            else:
                from model_utils.loss.pair_losses import EuclideanLoss
                criterion = EuclideanLoss(margin=1.0)
        elif args.model_type == 'triplet':
            if args.loss_type == 'cosine':
                from model_utils.loss.triplet_losses import CosineTripletLoss
                criterion = CosineTripletLoss(margin=0.1)
            elif args.loss_type == 'euclidean':
                from model_utils.loss.triplet_losses import EuclideanTripletLoss
                criterion = EuclideanTripletLoss(margin=1.0)
            elif args.loss_type == 'hybrid':
                from model_utils.loss.triplet_losses import HybridTripletLoss
                criterion = HybridTripletLoss(margin=1.0, alpha=0.5)
        elif args.model_type == 'supcon':
            if args.loss_type == 'supcon':
                from model_utils.loss.supcon_loss import SupConLoss
                criterion = SupConLoss(temperature=args.temperature)
            elif args.loss_type == 'infonce':
                from model_utils.loss.infonce_loss import InfoNCELoss
                criterion = InfoNCELoss(temperature=args.temperature)
        else:  # infonce
            from model_utils.loss.infonce_loss import InfoNCELoss
            criterion = InfoNCELoss(temperature=args.temperature)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        trainer = Trainer(model, criterion, optimizer, device)
        trainer.train(
            dataloader=None,  # You'll need to create appropriate dataloader
            test_reference_filepath=args.test_reference_filepath,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            epochs=args.epochs,
            medium_loader=None, 
            easy_loader = None,
            curriculum = args.curriculum
        )

    elif args.mode == 'grid_search':
        # Grid search
        if args.model_type == 'pair':
            model_class = SiameseCLIPModelPairs
        elif args.model_type == 'triplet':
            model_class = SiameseCLIPTriplet
        elif args.model_type == 'supcon':
            model_class = SiameseCLIPSupCon
        else:  # infonce
            model_class = SiameseCLIPInfoNCE
            
        searcher = GridSearcher(model_class, device, log_dir=args.log_dir)
        
        lrs = ast.literal_eval(args.lrs)
        batch_sizes = ast.literal_eval(args.batch_sizes)
        margins = ast.literal_eval(args.margins)
        internal_layer_sizes = ast.literal_eval(args.internal_layer_sizes)
        
        best_config, results_df = searcher.search(
            reference_filepath=args.reference_filepath,
            test_reference_filepath=args.test_reference_filepath,
            test_filepath=args.test_filepath,
            lrs=lrs,
            batch_sizes=batch_sizes,
            margins=margins,
            internal_layer_sizes=internal_layer_sizes,
            mode=args.model_type,
            loss_type=args.loss_type,
            medium_filepath=args.medium_filepath,
            easy_filepath=args.easy_filepath,
            epochs=args.epochs,
            temperature=args.temperature,
            curriculum=args.curriculum
        )
        
        print("\nGrid Search Results:")
        print(f"Best configuration: {best_config}")
        print("\nAll results saved to:", args.log_dir)

    elif args.mode in ['bayesian', 'random', 'optuna', 'pbt']:
        # Advanced hyperparameter optimization
        if args.model_type == 'pair':
            model_class = SiameseCLIPModelPairs
        elif args.model_type == 'triplet':
            model_class = SiameseCLIPTriplet
        elif args.model_type == 'supcon':
            model_class = SiameseCLIPSupCon
        else:  # infonce
            model_class = SiameseCLIPInfoNCE
        
        optimizer = UnifiedHyperparameterOptimizer(model_class, device, log_dir=args.log_dir)
        
        # Prepare optimization parameters
        opt_params = {
            'n_trials': args.n_trials,
            'n_calls': args.n_calls,
            'n_random_starts': args.n_random_starts,
            'population_size': args.population_size,
            'generations': args.generations,
            'epochs_per_generation': args.epochs_per_generation,
            'evolution_frequency': args.evolution_frequency,
            'sampler': args.sampler,
            'pruner': args.pruner if args.pruner != 'none' else None,
            'study_name': args.study_name,
            'epochs': args.epochs,
        }
        
        best_config, results_df, additional_info = optimizer.optimize(
            method=args.mode,
            reference_filepath=args.reference_filepath,
            test_reference_filepath=args.test_reference_filepath,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            loss_type=args.loss_type,
            medium_filepath=args.medium_filepath,
            easy_filepath=args.easy_filepath,
            **opt_params
        )
        
        print(f"\n{args.mode.upper()} Optimization Results:")
        print(f"Best configuration: {best_config}")
        print(f"Best accuracy: {best_config.get('best_accuracy', 'N/A')}")
        print(f"\nAll results saved to: {args.log_dir}")

    elif args.mode == 'compare':
        # Compare different optimization methods
        if args.model_type == 'pair':
            model_class = SiameseCLIPModelPairs
        elif args.model_type == 'triplet':
            model_class = SiameseCLIPTriplet
        elif args.model_type == 'supcon':
            model_class = SiameseCLIPSupCon
        else:  # infonce
            model_class = SiameseCLIPInfoNCE
        
        optimizer = UnifiedHyperparameterOptimizer(model_class, device, log_dir=args.log_dir)
        
        # Prepare optimization parameters
        opt_params = {
            'n_trials': args.n_trials,
            'n_calls': args.n_calls,
            'n_random_starts': args.n_random_starts,
            'epochs': args.epochs,
            'sampler': args.sampler,
            'pruner': args.pruner if args.pruner != 'none' else None
        }
        
        results = optimizer.compare_methods(
            reference_filepath=args.reference_filepath,
            test_reference_filepath=args.test_reference_filepath,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            loss_type=args.loss_type,
            medium_filepath=args.medium_filepath,
            easy_filepath=args.easy_filepath,
            **opt_params
        )
        
        print(f"\nComparison results saved to: {args.log_dir}")

if __name__ == '__main__':
    main() 