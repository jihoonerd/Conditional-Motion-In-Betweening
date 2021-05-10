def noise_injector(time_step: int, length: int):
    tta = length - time_step
    if tta < 5:
        noise_injection = 0.0
    elif 5 <= tta and tta < 30:
        noise_injection = (tta - 5) / 25.0
    else:
        noise_injection = 1.0
    return noise_injection
