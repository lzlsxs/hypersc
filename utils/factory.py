from models.hyperinversion import HyperInversion
def get_model(model_name, args):
    name = model_name.lower()
    if name == "hyperinversion":
        return HyperInversion(args)
    else:
        assert 0
