"""Cross-encoder reranker for v2's candidate pairs (Kaggle, 2x T4). One kernel per fold.

FOLD k trains on non-holdout train pairs with f2 != k and scores (out-of-fold) the train pairs with f2 == k.
FOLD 0 also scores the locked-holdout pairs (measurement only, never trained on) and all test pairs.
Input text per record: "name | address". Model: multilingual MiniLM cross-encoder (XLM-R vocabulary, covers Indic scripts).
Outputs in /kaggle/working: scores_<part>_<n>.parquet (rec, s1, xe = logit), model/ (weights), log lines on stdout.
A smoke pass (train 200 steps on 4k pairs, score 4k pairs) runs first so a bug fails in minutes, not hours.
"""
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FOLD = int("__FOLD__")
MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
N_TRAIN = int(os.environ.get("XE_N_TRAIN", 1_600_000))
MAX_LEN, BS_TRAIN, BS_SCORE, LR = 96, 128, 1024, 3e-5
OUT = "/kaggle/working"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


class Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def collate(tok):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return enc, torch.tensor(y, dtype=torch.float32)
    return f


def texts_for(pairs, text):
    return text.reindex(pairs.s1.values).fillna("").values, text.reindex(pairs.rec.values).fillna("").values


def train(model, tok, a, b, y, steps_cap=None):
    dl = DataLoader(Pairs(a, b, y), batch_size=BS_TRAIN, shuffle=True, num_workers=3, collate_fn=collate(tok),
                    drop_last=True, persistent_workers=False)
    total = min(len(dl), steps_cap or len(dl))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / total))
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    run = 0.0
    for step, (enc, yy) in enumerate(dl):
        if step >= total:
            break
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits[:, 0]
        loss = lossf(logits.float(), yy.cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % 1000 == 0:
            log(f"  step {step}/{total} loss {run:.4f}")


@torch.no_grad()
def score(model, tok, a, b):
    model.eval()
    dl = DataLoader(Pairs(a, b), batch_size=BS_SCORE, shuffle=False, num_workers=4, collate_fn=collate(tok))
    out = []
    for enc, _ in dl:
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def score_to_files(model, tok, pairs, text, part, chunk=2_000_000):
    for n, i in enumerate(range(0, len(pairs), chunk)):
        p = pairs.iloc[i:i + chunk]
        a, b = texts_for(p, text)
        xe = score(model, tok, a, b)
        pd.DataFrame({"rec": p.rec.values, "s1": p.s1.values, "xe": xe.astype(np.float32)}).to_parquet(
            f"{OUT}/scores_{part}_{n:02d}.parquet", index=False)
        log(f"  {part} chunk {n}: {len(p):,} pairs scored")


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def main():
    inp = os.path.dirname(glob.glob("/kaggle/input/**/pairs_train.parquet", recursive=True)[0])
    log(f"fold {FOLD}, input {inp}, GPUs {torch.cuda.device_count()}")
    fmt = lambda t: (t.name.fillna("").str.slice(0, 100) + " | " + t.addr.fillna("").str.slice(0, 140)).values
    tt = pd.read_parquet(f"{inp}/texts_train.parquet")
    text_tr = pd.Series(fmt(tt), index=tt.id.values); del tt
    pt = pd.read_parquet(f"{inp}/pairs_train.parquet")
    trn = pt[(~pt.ho) & (pt.f2 != FOLD)]
    trn = trn.sample(n=min(N_TRAIN, len(trn)), random_state=FOLD)
    oof = pt[(~pt.ho) & (pt.f2 == FOLD)]
    ho = pt[pt.ho]
    log(f"train sample {len(trn):,} (pos {trn.y.mean():.3f}) | oof {len(oof):,} | holdout {len(ho):,}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    # smoke pass: whole path on a few thousand pairs, then reload fresh weights
    s = trn.iloc[:4000]; a, b = texts_for(s, text_tr)
    train(net, tok, a, b, s.y.values.astype(np.float32), steps_cap=30)
    x = score(net, tok, a, b)
    log(f"smoke OK: {len(x)} scores, mean {x.mean():.3f}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    a, b = texts_for(trn, text_tr)
    log(f"training {math.ceil(len(trn) / BS_TRAIN)} steps")
    train(net, tok, a, b, trn.y.values.astype(np.float32))
    model.save_pretrained(f"{OUT}/model"); tok.save_pretrained(f"{OUT}/model")
    log("model saved")

    q = oof.sample(n=min(200_000, len(oof)), random_state=1)
    qa, qb = texts_for(q, text_tr)
    xs = score(net, tok, qa, qb)
    log(f"OOF sample AUC: cross-encoder {auc(q.y.values, xs):.5f} vs v2 p2 {auc(q.y.values, q.p2.values):.5f}")

    if FOLD == 0:
        score_to_files(net, tok, ho, text_tr, "holdout")
        hs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/scores_holdout_*.parquet"))])
        log(f"HOLDOUT pair AUC: cross-encoder {auc(ho.y.values, hs.xe.values):.5f} vs v2 p2 {auc(ho.y.values, ho.p2.values):.5f}")
    score_to_files(net, tok, oof, text_tr, "oof")
    del text_tr
    if FOLD == 0:
        tt = pd.read_parquet(f"{inp}/texts_test.parquet")
        text_te = pd.Series(fmt(tt), index=tt.id.values); del tt
        score_to_files(net, tok, pd.read_parquet(f"{inp}/pairs_test.parquet"), text_te, "test")
    log("DONE")


if __name__ == "__main__":
    main()
