import lightning.pytorch as L


def get_model_name(module: L.LightningModule) -> str:
    if not hasattr(module, 'model'):
        return module.__class__.__name__

    return module.model.__class__.__name__
