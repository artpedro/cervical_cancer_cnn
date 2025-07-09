import torch
from torch import nn
import timm
import torchvision.models as tvm


# Adapt model for binary classification
def _adapt_head(model: nn.Module, num_classes: int = 2) -> int:
    """
    Replace the model's final classification layer(s) with a fresh head that
    outputs `num_classes` logits.

    Parameters
    ----------
    model : nn.Module
        A CNN / ViT backbone (timm, torchvision, or custom).
    num_classes : int, default=2
        Desired number of output classes.

    Returns
    -------
    int
        The incoming feature dimension of the new head (`in_features`).

    Raises
    ------
    RuntimeError
        If no Linear or Conv2d head could be located.
    """

    # ---------------------------------------------------------------------
    # 1️⃣  FAST-PATH for *timm* models (ViT, EfficientNet, etc.)
    # ---------------------------------------------------------------------
    #   • timm exposes `reset_classifier`, which handles .head / .classifier
    #   • restoring this block lets us adapt any timm model in one call
    # ---------------------------------------------------------------------
    if hasattr(model, "reset_classifier"):
        old_head = model.get_classifier()  # works for both CNN & ViT
        in_feats = getattr(
            old_head, "in_features", getattr(old_head, "in_channels", None)
        )
        model.reset_classifier(num_classes)  # timm creates new nn.Linear
        return in_feats

    # ---------------------------------------------------------------------
    # 2️⃣  TorchVision & custom backbones
    # ---------------------------------------------------------------------
    #   • Added "head" to the list so ViTs without `reset_classifier`
    #     (or if someone deletes the fast-path) are still handled.
    # ---------------------------------------------------------------------
    for attr in ("head", "classifier", "fc", "_fc"):  # ← NEW: "head"
        if not hasattr(model, attr):
            continue

        head = getattr(model, attr)

        # --- Simple Linear head (e.g. ResNet, ViT) -----------------------
        if isinstance(head, nn.Linear):
            in_feats = head.in_features
            setattr(model, attr, nn.Linear(in_feats, num_classes))
            return in_feats

        # --- Sequential head (e.g. AlexNet, VGG) ------------------------
        if isinstance(head, nn.Sequential):
            layers = list(head.children())

            # Walk backwards to find the first Linear or Conv2d layer
            for idx in range(len(layers) - 1, -1, -1):
                layer = layers[idx]

                if isinstance(layer, nn.Linear):
                    in_feats = layer.in_features
                    layers[idx] = nn.Linear(in_feats, num_classes)
                    setattr(model, attr, nn.Sequential(*layers))
                    return in_feats

                if isinstance(layer, nn.Conv2d):
                    in_ch = layer.in_channels
                    layers[idx] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                    setattr(model, attr, nn.Sequential(*layers))
                    return in_ch

    # ---------------------------------------------------------------------
    # 3️⃣  Fallback
    # ---------------------------------------------------------------------
    raise RuntimeError("Could not find a Linear/Conv2d classification head.")


def load_any(name: str, num_classes: int = 2, pretrained: bool = True):
    """
    Load a backbone by name from timm, torchvision or PyTorch Hub and adapt it
    for `num_classes` outputs.
    """
    # timm
    try:
        model = timm.create_model(name, pretrained=pretrained)
        origin = f"timm:{name}"
    except (ValueError, RuntimeError):
        model, origin = None, None

    # torchvision
    if model is None:
        tv_registry = {
            "tv_squeezenet1_1": tvm.squeezenet1_1,
            "tv_shufflenet_v2_x1_0": tvm.shufflenet_v2_x1_0,
            "tv_mobilenet_v2": tvm.mobilenet_v2,
            "mobilenetv2_100": tvm.mobilenet_v2,
        }
        tv_ctor = tv_registry.get(name)
        if tv_ctor:
            weights = (
                tv_ctor.Weights.DEFAULT  # torchvision ≥0.15
                if pretrained and hasattr(tv_ctor, "Weights")
                else None
            )
            model = tv_ctor(weights=weights)
            origin = f"torchvision:{name}"

    # PyTorch
    if model is None and name.startswith("ghostnet"):
        model = torch.hub.load("pytorch/vision", "ghostnet_1x", pretrained=pretrained)
        origin = "hub:ghostnet"

    if model is None:
        raise ValueError(f"Unknown backbone: {name}")

    in_features = _adapt_head(model, num_classes)
    return model, in_features, origin
