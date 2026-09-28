from pathlib import Path


def classify_model(name: str, path: str | Path = "") -> str:
    """Return a conservative model family guess based on file and folder names."""
    haystack = f"{name} {path}".lower().replace("_", " ").replace("-", " ")
    if any(token in haystack for token in ("flux", " schnell", "dev model", "flux1")):
        return "flux"
    if any(token in haystack for token in ("sdxl", " xl ", "xl base", "juggernautxl", "ponydiffusion", "v6xl")):
        return "sdxl"
    if any(token in haystack for token in ("sd 1.5", "sd15", "v1 5", "1 5 pruned")):
        return "sd15"
    return "unknown"
