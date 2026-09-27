"""Backup bi-encoder blocking for ONE (split, country): multilingual-e5-small, name + address, max 40 tokens,
fp16 on both T4s; nearest neighbours by FAISS-GPU inner product (exact) if available, else torch matmul.
Output: /kaggle/working/nn_<split>_<country>_S<n>.parquet (s1_id, cand_id, cos, rank)."""
import glob, os, time, subprocess
import numpy as np, pandas as pd, torch
SPLIT, COUNTRY, K = "__SPLIT__", "__COUNTRY__", 10
t0 = time.time()
try:
    subprocess.run(["pip", "install", "-q", "faiss-gpu-cu12"], check=True, timeout=300)
    import faiss; HAVE_FAISS = faiss.get_num_gpus() > 0
except Exception as e:
    HAVE_FAISS = False; print("faiss unavailable:", e)
from sentence_transformers import SentenceTransformer
inp = glob.glob("/kaggle/input/**/texts_train.parquet", recursive=True)[0].rsplit("/", 1)[0]
d = pd.read_parquet(f"{inp}/texts_{SPLIT}.parquet")
d = d[d.country_norm == COUNTRY]
d["text"] = "query: " + d["business_name"].fillna("").str.slice(0, 80) + " | " + d["business_address"].fillna("").str.slice(0, 100)
print(f"[{time.time() - t0:.0f}s] {SPLIT}/{COUNTRY}: {len(d):,} records, faiss={HAVE_FAISS}", flush=True)
model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda"); model.half(); model.max_seq_length = 40
pool = model.start_multi_process_pool(["cuda:0", "cuda:1"])
def emb(texts):
    e = model.encode(texts, pool=pool, batch_size=1024, chunk_size=20000, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(e, dtype=np.float32)
q = d[d.src == 1]
Q = emb(q["text"].tolist()); print(f"[{time.time() - t0:.0f}s] S1 embedded {len(q):,}", flush=True)
for s in (2, 3):
    t = d[d.src == s]
    T = emb(t["text"].tolist()); print(f"[{time.time() - t0:.0f}s] S{s} embedded {len(t):,}", flush=True)
    if HAVE_FAISS:
        res = faiss.StandardGpuResources(); index = faiss.GpuIndexFlatIP(res, T.shape[1]); index.add(T)
        v, ix = index.search(Q, K)
    else:
        Tt = torch.from_numpy(T).half().cuda(); out_v, out_i = [], []
        for i in range(0, len(Q), 8192):
            sim = torch.from_numpy(Q[i:i + 8192]).half().cuda() @ Tt.T
            vv, ii = torch.topk(sim.float(), K, dim=1); out_v.append(vv.cpu().numpy()); out_i.append(ii.cpu().numpy())
        v, ix = np.vstack(out_v), np.vstack(out_i)
    ids = t["entity_id"].to_numpy()
    pd.DataFrame({"s1_id": np.repeat(q["entity_id"].to_numpy(), K), "cand_id": ids[ix.ravel()], "cos": v.ravel().astype(np.float32),
                  "rank": np.tile(np.arange(K, dtype=np.int8), len(q))}).to_parquet(f"/kaggle/working/nn_{SPLIT}_{COUNTRY}_S{s}.parquet", index=False)
    print(f"[{time.time() - t0:.0f}s] S{s} neighbours saved", flush=True)
model.stop_multi_process_pool(pool)
print(f"DONE {SPLIT}/{COUNTRY} in {time.time() - t0:.0f}s")
