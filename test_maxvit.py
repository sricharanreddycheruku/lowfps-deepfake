from torchvision.models import maxvit_t, MaxVit_T_Weights
try:
    m = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)
    print("MAXVIT_OK", sum(p.numel() for p in m.parameters()))
except Exception as e:
    print("MAXVIT_ERR", e)
