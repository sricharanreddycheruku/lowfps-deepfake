import timm
try:
    m = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    print("DINO_OK", sum(p.numel() for p in m.parameters()))
except Exception as e:
    print("DINO_ERR", e)
