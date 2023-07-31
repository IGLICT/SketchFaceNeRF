def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    #print(model_type, " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if model_type == "pixelnerf":
        from .models import PixelNeRFNet
        net = PixelNeRFNet(conf, *args, **kwargs)
    elif model_type == "planes_eg3d":
        print("use eg3d planes")
        from .models_planes_eg3d import PixelNeRFNet
        net = PixelNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
