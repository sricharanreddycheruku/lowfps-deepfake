import timm
for arch in ["tf_efficientnetv2_s.in21k_ft_in1k", "vit_small_patch16_224.dino"]:
    try:
        m = timm.create_model(arch, pretrained=True)
        print(f"SUCCESS: {arch} loaded with {sum(p.numel() for p in m.parameters()):,} parameters")
    except Exception as e:
        print(f"FAILED: {arch} -> {e}")
