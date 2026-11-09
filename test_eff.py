import timm
m = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True)
print("EFFV2S_OK", sum(p.numel() for p in m.parameters()))
