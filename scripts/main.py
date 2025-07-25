import argparse
import ast
import torch
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from scripts.grid_search.grid_searcher import GridSearcher
from scripts.baseline.baseline_tester import BaselineTester
from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer
from model_utils.models.learning.siamese import SiameseModelPairs, SiameseModelTriplet
from model_utils.models.learning.supcon import SiameseModelSupCon
from model_utils.models.learning.infonce import SiameseModelInfoNCE

def main():
    parser = argparse.ArgumentParser(description='CLIP-based text similarity training and evaluation')
    parser.add_argument('--mode', type=str, 
                      choices=['train', 'grid_search', 'bayesian', 'random', 'optuna', 'compare', 'baseline', 'evaluate_saved'], 
                      required=True,
                      help='Mode to run: train, grid_search, bayesian, random, optuna, compare, or baseline (supports multiple vision-language models)')
    parser.add_argument('--training_filepath', type=str,
                      help='Path to training data (for training modes)')
    parser.add_argument('--test_filepath', type=str, required=True,
                      help='Path to test data (CSV or Parquet with fraudulent_name, real_name, label)')
    parser.add_argument('--model_type', type=str, choices=['pair', 'triplet', 'supcon', 'infonce'], default='pair',
                      help='Model type: pair, triplet, supcon, or infonce')
    parser.add_argument('--loss_type', type=str, choices=['cosine', 'euclidean', 'hybrid', 'supcon', 'infonce'], default='cosine',
                      help='Loss function type')
    parser.add_argument('--baseline_model', type=str, choices=['clip', 'coca', 'flava', 'align', 'all'], default='clip',
                      help='Baseline model to test (for baseline mode)')
    parser.add_argument('--backbone', type=str, choices=['clip', 'coca', 'flava', 'siglip'], default='clip',
                      help='Vision-language backbone to use (clip, siglip, flava, etc.)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--medium_filepath', type=str,
                      help='Path to medium data (optional)')
    parser.add_argument('--easy_filepath', type=str,
                      help='easy to medium data (optional)')
    parser.add_argument('--external', action='store_true', default=False,
                      help='If set, evaluate on an external pairwise dataset (no reference set, only test_filepath required)')
    parser.add_argument('--validate_filepath', type=str, default=None, help='Path to validation data file (CSV or Parquet). Used for mid-training and end-of-training validation.')
    parser.add_argument('--plot', action='store_true', help='If set, plot ROC and confusion matrices during evaluation')
    
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
    parser.add_argument('--sampler', type=str, choices=['tpe', 'random', 'cmaes'], default='tpe',
                      help='Sampler for Optuna optimization')
    parser.add_argument('--pruner', type=str, choices=['median', 'hyperband', 'none'], default='median',
                      help='Pruner for Optuna optimization')
    parser.add_argument('--study_name', type=str,
                      help='Study name for Optuna optimization')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_siamese_model(mode, backbone_name, embedding_dim=512, projection_dim=128, device=None):
        from scripts.baseline.baseline_tester import BaselineTester
        tester = BaselineTester(model_type=backbone_name, batch_size=1, device=device)
        backbone_module = tester.model_wrapper  # Use the wrapper, not .model
        assert hasattr(backbone_module, 'encode_text'), f"Backbone {type(backbone_module)} does not have encode_text"
        if mode == 'pair':
            return SiameseModelPairs(embedding_dim, projection_dim, backbone=backbone_module)
        elif mode == 'triplet':
            return SiameseModelTriplet(embedding_dim, projection_dim, backbone=backbone_module)
        elif mode == 'supcon':
            return SiameseModelSupCon(embedding_dim, projection_dim, backbone=backbone_module)
        elif mode == 'infonce':
            return SiameseModelInfoNCE(embedding_dim, projection_dim, backbone=backbone_module)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if args.mode == 'baseline':
        from scripts.baseline.baseline_tester import BaselineTester
        # Test baseline model(s) performance (pairwise only)
        if args.baseline_model == 'all':
            print("Testing all available baseline models...")
            tester = BaselineTester(model_type='clip', batch_size=args.batch_size, device=device)
            all_results = tester.test_all_models(args.test_filepath)
            print("\nBaseline Results Summary:")
            for model_type, result in all_results.items():
                if 'error' in result:
                    print(f"{model_type.upper()}: ERROR - {result['error']}")
                else:
                    metrics = result['metrics']
                    # Print only relevant metrics
                    metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
                    print(f"{model_type.upper()}: {metrics_to_print}")
        else:
            print(f"Testing {args.baseline_model.upper()} baseline model...")
            tester = BaselineTester(model_type=args.baseline_model, batch_size=args.batch_size, device=device)
            results_df, metrics = tester.test(args.test_filepath, plot=args.plot)
            print(f"\n{args.baseline_model.upper()} Baseline Results:")
            metrics_to_print = {k: v for k, v in metrics.items() if k != 'roc_curve'}
            print(metrics_to_print)
    elif args.mode == 'evaluate_saved':
        print("Loading saved model for evaluation...")
        # Load backbone
        from scripts.baseline.baseline_tester import BaselineTester
        tester = BaselineTester(model_type=args.backbone, batch_size=1, device=device)
        backbone_module = tester.model_wrapper  # must have .encode_text
        
        # Load your model with matching dimensions
        model = SiameseModelPairs(embedding_dim=768, projection_dim=512, backbone=backbone_module).to(device)

        # Load saved weights
        state_dict = torch.load(args.log_dir + "/best_model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Evaluate
        evaluator = Evaluator(model, batch_size=args.batch_size, model_type=args.model_type)
        results_df, metrics = evaluator.evaluate(args.test_filepath, plot=args.plot)

        print("\n Evaluation complete. Results:")
        for k, v in metrics.items():
            if k != 'roc_curve':
                print(f"{k}: {v}")

    elif args.mode == 'train':
        # Single training run
        model = get_siamese_model(args.model_type, args.backbone, embedding_dim=512, projection_dim=128, device=device).to(device)
        
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
        
        # Create dataloaders
        import pandas as pd
        from torch.utils.data import DataLoader
        
        # Load training data
        dataframe = pd.read_parquet(args.training_filepath)
        
        # Create appropriate dataset and dataloader based on model type
        if args.model_type == "pair":
            from utils.data import TextPairDataset
            dataset = TextPairDataset(dataframe)
        elif args.model_type == "triplet":
            from utils.data import TripletDataset
            dataset = TripletDataset(dataframe)
        elif args.model_type == "supcon":
            from utils.data import SupConDataset
            dataset = SupConDataset(dataframe)
        elif args.model_type == "infonce":
            from utils.data import InfoNCEDataset
            dataset = InfoNCEDataset(dataframe)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        # Create warmup dataloader if warmup filepath is provided
        easy_loader = None
        medium_loader = None
        if args.easy_filepath and args.medium_filepath:
            medium_dataframe = pd.read_parquet(args.medium_filepath)
            easy_dataframe = pd.read_parquet(args.easy_filepath)
            
            if args.model_type == "pair":
                from utils.data import TextPairDataset
                medium_dataset = TextPairDataset(medium_dataframe)
                easy_dataset = TextPairDataset(easy_dataframe)
            elif args.model_type == "triplet":
                from utils.data import TripletDataset
                medium_dataset = TripletDataset(medium_dataframe)
                easy_dataset = TripletDataset(easy_dataframe)
            elif args.model_type == "supcon":
                from utils.data import SupConDataset
                medium_dataset = SupConDataset(medium_dataframe)
                easy_dataset = SupConDataset(easy_dataframe)
            elif args.model_type == "infonce":
                from utils.data import InfoNCEDataset
                medium_dataset = InfoNCEDataset(medium_dataframe)
                easy_dataset = InfoNCEDataset(easy_dataframe)
            else:
                raise ValueError(f"Unknown model type: {args.model_type}")
            
            medium_loader = DataLoader(medium_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            easy_loader = DataLoader(easy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        ### here: pass in the model_type
        trainer = Trainer(model, criterion, optimizer, device, model_type=args.model_type)
        trainer.train(
            dataloader=dataloader,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            epochs=args.epochs,
            medium_loader=medium_loader,
            easy_loader=easy_loader,
            curriculum=args.curriculum,
            validate_filepath=args.validate_filepath,
            plot=args.plot
        )

    elif args.mode == 'grid_search':
        # Grid search
        from scripts.baseline.baseline_tester import BaselineTester
        tester = BaselineTester(model_type=args.backbone, batch_size=1, device=device)
        backbone_module = tester.model_wrapper
        def model_class_factory(embedding_dim, projection_dim):
            # Debug print statements
            if isinstance(embedding_dim, (tuple, list)):
                embedding_dim = embedding_dim[0]
            if isinstance(projection_dim, (tuple, list)):
                projection_dim = projection_dim[0]
            # Always cast to int
            embedding_dim = int(embedding_dim)
            projection_dim = int(projection_dim)
            return get_siamese_model(args.model_type, args.backbone, embedding_dim=embedding_dim, projection_dim=projection_dim, device=device)
        searcher = GridSearcher(model_class_factory, device, log_dir=args.log_dir, backbone=backbone_module)
        
        lrs = ast.literal_eval(args.lrs)
        batch_sizes = ast.literal_eval(args.batch_sizes)
        margins = ast.literal_eval(args.margins)
        internal_layer_sizes = ast.literal_eval(args.internal_layer_sizes)
        
        best_config, results_df = searcher.search(
            training_filepath=args.training_filepath,
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
            curriculum=args.curriculum,
            validate_filepath=args.validate_filepath
        )
        
        print("\nGrid Search Results:")
        print(f"Best configuration: {best_config}")
        print("\nAll results saved to:", args.log_dir)

    elif args.mode in ['bayesian', 'random', 'optuna']:
        # Advanced hyperparameter optimization
        optimizer = UnifiedHyperparameterOptimizer(args.backbone, device=device, log_dir=args.log_dir)
        
        # Prepare optimization parameters
        opt_params = {
            'n_trials': args.n_trials,
            'n_calls': args.n_calls,
            'n_random_starts': args.n_random_starts,
            'sampler': args.sampler,
            'pruner': args.pruner if args.pruner != 'none' else None,
            'study_name': args.study_name,
            'epochs': args.epochs,
        }
        
        results = optimizer.optimize(
            method=args.mode,
            training_filepath=args.training_filepath,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            loss_type=args.loss_type,
            medium_filepath=args.medium_filepath,
            easy_filepath=args.easy_filepath,
            **opt_params,
            validate_filepath=args.validate_filepath
        )

    elif args.mode == 'compare':
        # Compare different optimization methods
        optimizer = UnifiedHyperparameterOptimizer(args.backbone, device=device, log_dir=args.log_dir)
        
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
            training_filepath=args.training_filepath,
            test_filepath=args.test_filepath,
            mode=args.model_type,
            loss_type=args.loss_type,
            medium_filepath=args.medium_filepath,
            easy_filepath=args.easy_filepath,
            **opt_params,
            validate_filepath=args.validate_filepath
        )
        
        print(f"\nComparison results saved to: {args.log_dir}")

if __name__ == '__main__':
    main() 
