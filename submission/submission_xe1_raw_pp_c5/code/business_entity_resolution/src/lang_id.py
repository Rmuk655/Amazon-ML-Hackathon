"""IndicLID (native-script fastText model, "FTN") wrapper.

We only need the native-script classifier: Latin-script text is never sent to
transliteration, so the heavy Roman IndicBERT path of IndicLID is skipped.

Setup: download indiclid-ftn.zip from https://github.com/AI4Bharat/IndicLID/releases
and unzip into  code/business_entity_resolution/models/indiclid-ftn/
(any *.bin inside is picked up). If the model is missing we silently fall back to
script-default languages, so the pipeline still runs.
"""
from pathlib import Path

from config import INDICLID_FTN_DIR, LID_CONFIDENCE_THRESHOLD

# IndicLID label prefix -> IndicXlit language code
LID_TO_XLIT = {
    "asm": "as", "ben": "bn", "brx": "brx", "guj": "gu", "hin": "hi", "kan": "kn",
    "kas": "ks", "kok": "gom", "mai": "mai", "mal": "ml", "mni": "mni", "mar": "mr",
    "nep": "ne", "ori": "or", "pan": "pa", "san": "sa", "snd": "sd", "tam": "ta",
    "tel": "te", "urd": "ur",
}


class NativeLID:
    def __init__(self, model_dir=INDICLID_FTN_DIR, threshold: float = LID_CONFIDENCE_THRESHOLD, batch: int = 50_000):
        self.threshold, self.batch, self.model = threshold, batch, None
        model_dir = Path(model_dir)
        bins = sorted(model_dir.glob("**/*.bin")) if model_dir.exists() else []
        if bins:
            import fasttext  # pip install fasttext-wheel  (needs numpy<2)
            self.model = fasttext.load_model(str(bins[0]))
            print(f"[lang_id] loaded {bins[0]}")
        else:
            print(f"[lang_id] no IndicLID model in {model_dir} -> using script-default languages")

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, texts):
        """-> list of (label like 'hin_Deva' or None, confidence)."""
        if self.model is None:
            return [(None, 0.0)] * len(texts)
        out = []
        for i in range(0, len(texts), self.batch):
            labels, probs = self.model.predict(texts[i:i + self.batch], k=1)
            out += [(l[0].replace("__label__", ""), float(p[0])) for l, p in zip(labels, probs)]
        return out

    def resolve(self, label, conf, script):
        """IndicXlit code if the prediction is confident, supported and script-consistent."""
        if not label or conf < self.threshold:
            return None
        lang, _, lscript = label.partition("_")
        lscript = "Taml" if lscript == "Tamil" else lscript
        if lscript != script:
            return None
        return LID_TO_XLIT.get(lang)