class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


model_configs = dotdict({
	"checkpoint_name": 'mnist-classifier-with-augmentation',
	"save_dir": '../saved_models/',
    "data_dir": '../data/',
    "result_dir": '../results/',
    "test_ratio": 0.1,
    "val_ratio": 0.1,
    "num_workers" : 8,     
    "img_size": 32,
    "in_channels": 3,
    "num_classes": 10,
    "num_layers": 2,
    "embed_channels" : 128,
    "hidden_channels" : 256,
    "random_seed" : 42
})

training_configs = dotdict({
    "num_epochs": 20,
	"learning_rate": 1e-3,
    "batch_size": 64,
    "save_interval": 10,
    "weight_decay": 1e-5
})
