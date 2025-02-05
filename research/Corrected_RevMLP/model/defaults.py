def model_defaults():
    "Defaults for Dense Model Training"
    res = {
        "Model": {
            "feature_dim": 3072,
            "hidden_dim": 2800,
            "output_dim": 10,
            "bias": True,
            "num_layers": 6,
            "layer_norm": False,
            "non_linearity": True,
            "rtol": 1e-3,
            "atol": 1e-6,
        }
    }
    return res
