"""
Sistem Rekomendasi Buku - Book-Crossing (BX)
Streamlit app berdasarkan notebook tugas.

Metode : Popularity (weighted rating), Item-based CF (kNN cosine),
         SVD (matrix factorization), Content-based (TF-IDF judul/penulis/penerbit)
Evaluasi: RMSE/MAE & Precision/Recall/NDCG/HitRate@10

Jalankan:  streamlit run app.py
Data    :  taruh BX-Book-Ratings.csv, BX-Users.csv, BX-Books.csv di folder `data/`
           (atau upload lewat sidebar).
"""
import csv
import io
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Rekomendasi Buku BX", page_icon="📚", layout="wide")

METHODS = ["Popularity", "Item-kNN", "SVD", "Content-based"]


# ---------------------------------------------------------------- 1. LOAD
def _read_books(src):
    raw = pd.read_csv(src, sep=";", encoding="latin-1", escapechar="\\",
                      on_bad_lines="skip", quoting=csv.QUOTE_NONE, dtype=str)
    raw.columns = raw.columns.str.strip('"')
    books = raw[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]].copy()
    books.columns = ["ISBN", "Title", "Author", "Year", "Publisher"]
    for c in books.columns:
        books[c] = books[c].astype(str).str.strip('"')
    return books


@st.cache_data(show_spinner="Membaca CSV...")
def load_data(rat_src, books_src):
    """rat_src / books_src: path (str) atau bytes hasil upload."""
    wrap = lambda s: io.BytesIO(s) if isinstance(s, (bytes, bytearray)) else s
    rat = pd.read_csv(wrap(rat_src), sep=";", encoding="latin-1", dtype={"ISBN": str})
    books = _read_books(wrap(books_src))
    return rat, books


# ------------------------------------------------- 2-4. CLEAN, SPLIT, MODEL
@st.cache_resource(show_spinner="Cleaning data & melatih model...")
def build(rat, books, min_cnt=5, test_frac=0.2, K=30, F=40, seed=42):
    rat = rat.copy(); books = books.copy()
    rat["ISBN"] = rat["ISBN"].astype(str).str.strip().str.upper()
    books["ISBN"] = books["ISBN"].str.strip().str.upper()
    books = books.drop_duplicates("ISBN")
    rat = rat[rat["ISBN"].isin(books["ISBN"])]
    implicit_share = float((rat["Book-Rating"] == 0).mean())

    ex = rat[rat["Book-Rating"] > 0].rename(columns={"User-ID": "u", "ISBN": "i", "Book-Rating": "r"})
    for _ in range(3):
        ex = ex[ex.groupby("u")["r"].transform("size") >= min_cnt]
        ex = ex[ex.groupby("i")["r"].transform("size") >= min_cnt]
    if ex.empty:
        raise ValueError("Data kosong setelah filter. Turunkan 'minimal rating'.")

    uid = {u: k for k, u in enumerate(ex.u.unique())}
    iid = {i: k for k, i in enumerate(ex.i.unique())}
    ex = ex.copy(); ex["ui"] = ex.u.map(uid); ex["ii"] = ex.i.map(iid)
    inv_i = {v: k for k, v in iid.items()}; inv_u = {v: k for k, v in uid.items()}
    nU, nI = len(uid), len(iid)
    binfo = books.set_index("ISBN")

    # split 20% rating tiap user -> test
    ex = ex.sample(frac=1, random_state=seed)
    ex["rank"] = ex.groupby("ui").cumcount(); ex["n"] = ex.groupby("ui")["r"].transform("size")
    test = ex[ex["rank"] < np.maximum(1, (ex.n * test_frac).astype(int))]
    train = ex.drop(test.index)
    R = sp.csr_matrix((train.r.values.astype(np.float32), (train.ui, train.ii)), shape=(nU, nI))
    gm = train.r.mean()

    # bias baseline
    g = train.groupby("ii").r
    ib = ((g.sum() - gm * g.size()) / (g.size() + 10)).reindex(range(nI)).fillna(0).values
    tt = train.assign(res=train.r - gm - ib[train.ii])
    gu = tt.groupby("ui").res
    ub = (gu.sum() / (gu.size() + 10)).reindex(range(nU)).fillna(0).values

    # popularity (weighted rating)
    cnt = np.asarray((R > 0).sum(0)).ravel()
    mean_i = np.asarray(R.sum(0)).ravel() / np.maximum(cnt, 1)
    m = np.percentile(cnt, 75)
    pop_score = cnt / (cnt + m) * mean_i + m / (cnt + m) * gm

    # item-kNN (cosine pada rating terpusat)
    Rc = R.copy().tocoo(); Rc.data = Rc.data - gm - ub[Rc.row]; Rc = Rc.tocsr()
    Rn = normalize(Rc.T.tocsr(), axis=1)
    S = (Rn @ Rn.T).tocsr(); S.setdiag(0); S.eliminate_zeros()
    rows, cols, vals = [], [], []
    for a in range(S.shape[0]):
        s, e = S.indptr[a], S.indptr[a + 1]; d = S.data[s:e]; c = S.indices[s:e]
        if len(d) > K:
            idx = np.argpartition(-d, K)[:K]; d, c = d[idx], c[idx]
        rows += [a] * len(d); cols += list(c); vals += list(d)
    S = sp.csr_matrix((vals, (rows, cols)), shape=S.shape)
    S.data = np.maximum(S.data, 0); S.eliminate_zeros()
    B = (R > 0).astype(np.float32)

    # SVD pada residual bias
    Rres = R.copy().tocoo(); Rres.data = Rres.data - gm - ub[Rres.row] - ib[Rres.col]; Rres = Rres.tocsr()
    F = int(min(F, min(nU, nI) - 1))
    U_, s_, Vt_ = svds(Rres.astype(np.float64), k=F)
    P = U_ * np.sqrt(s_); Q = Vt_.T * np.sqrt(s_)

    # content-based TF-IDF
    meta = books.set_index("ISBN").reindex([inv_i[k] for k in range(nI)])
    txt = (meta.Title.fillna("") + " " + meta.Author.fillna("").str.replace(" ", "_") + " "
           + meta.Publisher.fillna("").str.replace(" ", "_")).str.lower()
    X = normalize(TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2),
                                  max_features=100000).fit_transform(txt))

    stats = dict(n_rating=len(ex), n_user=nU, n_book=nI, n_train=len(train), n_test=len(test),
                 sparsity=1 - len(ex) / (nU * nI), mean=ex.r.mean(), implicit=implicit_share,
                 dist=ex.r.value_counts().sort_index(), item_sim_nnz=S.nnz)
    return dict(uid=uid, iid=iid, inv_i=inv_i, inv_u=inv_u, nU=nU, nI=nI, binfo=binfo,
                train=train, test=test, R=R, gm=gm, ib=ib, ub=ub, cnt=cnt, pop_score=pop_score,
                Rc=Rc, S=S, B=B, P=P, Q=Q, X=X, stats=stats)


# ------------------------------------------------------------- SCORING
def scores_for(M, method, users):
    R, S, B = M["R"], M["S"], M["B"]
    Ru = R[users]
    if method == "Popularity":
        sc = np.tile(M["pop_score"], (len(users), 1))
    elif method == "Item-kNN":
        sc = (M["Rc"][users] @ S.T).toarray() / (np.asarray((B[users] @ S.T).todense()) + 1) \
             + 0.01 * M["pop_score"]
    elif method == "SVD":
        sc = M["P"][users] @ M["Q"].T + M["ib"][None, :]
    else:  # Content-based
        prof = normalize(Ru.multiply(Ru >= 7) @ M["X"])
        sc = (prof @ M["X"].T).toarray()
    sc[Ru.nonzero()] = -np.inf
    return sc


def scores_from_items(M, items, ratings, method):
    """Cold-start: skor untuk user baru dari daftar buku favorit."""
    nI = M["nI"]
    v = sp.csr_matrix((np.asarray(ratings, np.float32), ([0] * len(items), items)), shape=(1, nI))
    if method == "Item-kNN":
        c = v.copy(); c.data = c.data - M["gm"]
        num = (c @ M["S"].T).toarray()[0]
        den = ((v > 0).astype(np.float32) @ M["S"].T).toarray()[0] + 1
        sc = num / den + 0.01 * M["pop_score"]
    elif method == "Content-based":
        prof = normalize(v.multiply(v >= 7) @ M["X"])
        sc = (prof @ M["X"].T).toarray()[0]
    else:  # SVD fold-in
        res = np.asarray(ratings) - M["gm"] - M["ib"][items]
        Qi = M["Q"][items]
        p = np.linalg.solve(Qi.T @ Qi + 0.1 * np.eye(Qi.shape[1]), Qi.T @ res)
        sc = M["Q"] @ p + M["ib"]
    sc[items] = -np.inf
    return sc


# ------------------------------------------------------------ EVALUASI
@st.cache_data(show_spinner="Menghitung RMSE/MAE...")
def eval_rating(_M, key):
    M = _M; t = M["test"]
    tu, ti, tr = t.ui.values, t.ii.values, t.r.values
    base = M["gm"] + M["ub"][tu] + M["ib"][ti]
    Rc, S, B = M["Rc"], M["S"], M["B"]
    num = np.asarray((Rc[tu].multiply(S[ti])).sum(1)).ravel()   # = (Rc @ S.T)[u,i]
    den = np.asarray((B[tu].multiply(S[ti])).sum(1)).ravel()
    knn = base + np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1), 0)
    svd = base + (M["P"][tu] * M["Q"][ti]).sum(1)
    preds = {"Global mean": np.full_like(tr, M["gm"], dtype=float), "Bias baseline": base,
             "Item-kNN": knn, "SVD": svd}
    out = {}
    for k, p in preds.items():
        p = np.clip(p, 1, 10)
        out[k] = dict(RMSE=np.sqrt(np.mean((p - tr) ** 2)), MAE=np.mean(np.abs(p - tr)))
    return pd.DataFrame(out).T


@st.cache_data(show_spinner="Menghitung metrik Top-N...")
def eval_topn(_M, key, N=10, thr=7, n_users=3000, seed=42):
    M = _M; test = M["test"]
    rng = np.random.default_rng(seed)
    tu = np.unique(test.ui.values)
    users = rng.choice(tu, size=min(n_users, len(tu)), replace=False)
    rel = test[test.r >= thr].groupby("ui").ii.apply(set).to_dict()
    users = np.array([u for u in users if u in rel])
    out = {}
    disc = 1 / np.log2(np.arange(2, N + 2))
    for meth in METHODS:
        Pp, Rr, ND, hit = [], [], [], []
        for b in range(0, len(users), 500):
            us = users[b:b + 500]; sc = scores_for(M, meth, us)
            top = np.argpartition(-sc, N, axis=1)[:, :N]
            for k, u in enumerate(us):
                t = top[k][np.argsort(-sc[k, top[k]])]; g = rel[u]
                h = np.array([x in g for x in t])
                Pp.append(h.mean()); Rr.append(h.sum() / len(g)); hit.append(h.any())
                ND.append((h * disc).sum() / disc[:min(len(g), N)].sum())
        out[meth] = {f"Precision@{N}": np.mean(Pp), f"Recall@{N}": np.mean(Rr),
                     f"NDCG@{N}": np.mean(ND), f"HitRate@{N}": np.mean(hit)}
    return pd.DataFrame(out).T, len(users)


# ------------------------------------------------------------ HELPERS
def book_table(M, idx, score=None):
    b, inv = M["binfo"], M["inv_i"]
    rows = []
    for j, i in enumerate(idx):
        isbn = inv[i]
        r = dict(ISBN=isbn, Judul=b.Title.get(isbn), Penulis=b.Author.get(isbn),
                 Tahun=b.Year.get(isbn), Penerbit=b.Publisher.get(isbn))
        if score is not None:
            r["Skor"] = round(float(score[j]), 4)
        rows.append(r)
    df = pd.DataFrame(rows); df.index = np.arange(1, len(df) + 1)
    return df


def top_n(sc, n, dedup_title=False, M=None):
    order = np.argsort(-sc)
    order = order[np.isfinite(sc[order])]
    if not dedup_title:
        return order[:n]
    seen, keep = set(), []
    for i in order:
        t = str(M["binfo"].Title.get(M["inv_i"][i])).lower()
        if t in seen:
            continue
        seen.add(t); keep.append(i)
        if len(keep) == n:
            break
    return np.array(keep, dtype=int)


# ================================================================ UI
st.title("📚 Sistem Rekomendasi Buku — Book-Crossing")
st.caption("Popularity · Item-based CF (kNN cosine) · SVD · Content-based (TF-IDF)")

with st.sidebar:
    st.header("1. Data")
    src = st.radio("Sumber data", ["Folder lokal", "Upload CSV"], horizontal=True)
    if src == "Folder lokal":
        folder = st.text_input("Folder", "data")
        rp, bp = os.path.join(folder, "BX-Book-Ratings.csv"), os.path.join(folder, "BX-Books.csv")
        ok = os.path.exists(rp) and os.path.exists(bp)
        rat_src, books_src = (rp, bp) if ok else (None, None)
        if not ok:
            st.warning(f"File BX-Book-Ratings.csv / BX-Books.csv tidak ditemukan di `{folder}/`.")
    else:
        fr = st.file_uploader("BX-Book-Ratings.csv", type="csv")
        fb = st.file_uploader("BX-Books.csv", type="csv")
        rat_src = fr.getvalue() if fr else None
        books_src = fb.getvalue() if fb else None

    st.header("2. Parameter")
    min_cnt = st.number_input("Minimal rating per user & per buku", 2, 50, 5)
    test_frac = st.slider("Porsi test per user", 0.1, 0.4, 0.2, 0.05)
    K = st.slider("K tetangga (Item-kNN)", 5, 100, 30, 5)
    F = st.slider("Faktor laten (SVD)", 5, 100, 40, 5)
    dedup = st.checkbox("Sembunyikan judul duplikat (beda ISBN)", True)

if rat_src is None or books_src is None:
    st.info("Siapkan data dulu di sidebar. Unduh dataset Book-Crossing (BX-Book-Ratings.csv, "
            "BX-Books.csv), lalu taruh di folder `data/` atau upload.")
    st.stop()

rat, books = load_data(rat_src, books_src)
try:
    M = build(rat, books, int(min_cnt), float(test_frac), int(K), int(F))
except ValueError as e:
    st.error(str(e)); st.stop()
key = (int(min_cnt), float(test_frac), int(K), int(F), len(rat))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Data", "📈 Evaluasi", "👤 Rekomendasi User", "✨ User Baru", "🔗 Buku Mirip", "🔥 Populer"])

# ---------------- Data
with tab1:
    s = M["stats"]
    c = st.columns(4)
    c[0].metric("Rating (eksplisit)", f"{s['n_rating']:,}")
    c[1].metric("User", f"{s['n_user']:,}")
    c[2].metric("Buku", f"{s['n_book']:,}")
    c[3].metric("Sparsity", f"{s['sparsity']*100:.4f}%")
    c = st.columns(4)
    c[0].metric("Rata-rata rating", f"{s['mean']:.2f}")
    c[1].metric("Porsi rating implisit (0)", f"{s['implicit']*100:.1f}%")
    c[2].metric("Train / Test", f"{s['n_train']:,} / {s['n_test']:,}")
    c[3].metric("Item-sim nnz", f"{s['item_sim_nnz']:,}")
    st.caption("Rating 0 (implisit) dibuang; hanya rating eksplisit 1–10 yang dipakai. "
               "Filter user & buku dilakukan iteratif 3x.")
    st.subheader("Distribusi rating")
    st.bar_chart(s["dist"].rename("jumlah"))
    st.subheader("Contoh data buku")
    st.dataframe(books.head(20))

# ---------------- Evaluasi
with tab2:
    st.subheader("Prediksi rating (RMSE / MAE)")
    st.dataframe(eval_rating(M, key).round(4))
    st.subheader("Top-10 (relevan = rating test ≥ 7)")
    if st.button("Hitung metrik Top-N", type="primary"):
        st.session_state["topn"] = key
    if st.session_state.get("topn") == key:
        topn, nu = eval_topn(M, key)
        st.caption(f"User uji: {nu}")
        st.dataframe(topn.round(4))
        st.bar_chart(topn)
        allres = pd.concat([eval_rating(M, key), topn], axis=1).round(4)
        st.download_button("⬇️ Unduh hasil_evaluasi.csv", allres.to_csv().encode(),
                           "hasil_evaluasi.csv", "text/csv")
    else:
        st.caption("Klik tombol di atas (butuh beberapa detik–menit tergantung ukuran data).")

# ---------------- Rekomendasi user lama
with tab3:
    tr = M["train"]
    by_cnt = tr.groupby("ui").size().sort_values(ascending=False)
    opts = [M["inv_u"][u] for u in by_cnt.index]
    c1, c2, c3 = st.columns([2, 2, 1])
    user = c1.selectbox("User-ID (urut dari yang paling aktif)", opts,
                        index=min(5, len(opts) - 1))
    meth = c2.selectbox("Metode", METHODS, index=2)
    n = c3.number_input("Jumlah", 5, 50, 10)
    u = M["uid"][user]
    hist = tr[tr.ui == u].sort_values("r", ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Riwayat rating ({len(hist)} buku, 10 teratas)**")
        h = book_table(M, hist.ii.values[:10]); h["Rating"] = hist.r.values[:10]
        st.dataframe(h[["Judul", "Penulis", "Rating"]])
    with right:
        st.markdown(f"**Rekomendasi {meth}**")
        sc = scores_for(M, meth, [u])[0]
        top = top_n(sc, int(n), dedup, M)
        st.dataframe(book_table(M, top, sc[top])[["Judul", "Penulis", "Skor"]],
                     )

# ---------------- User baru (cold start)
with tab4:
    st.markdown("Pilih beberapa buku yang kamu suka, beri rating, lalu dapatkan rekomendasi.")
    pop_idx = [int(x) for x in np.argsort(-M["cnt"])]
    labels = {i: f"{M['binfo'].Title.get(M['inv_i'][i])} — {M['binfo'].Author.get(M['inv_i'][i])}"
              for i in pop_idx}
    lab2idx = {}
    for i, l in labels.items():
        lab2idx.setdefault(l, i)          # judul+penulis sama -> ambil yang paling populer
    picked = st.multiselect("Buku favorit", list(lab2idx.keys()), max_selections=15)
    picks = [lab2idx[l] for l in picked]
    if picks:
        rts = [st.slider(labels[i][:80], 1, 10, 9, key=f"r{i}") for i in picks]
        c1, c2 = st.columns(2)
        meth2 = c1.selectbox("Metode ", ["Item-kNN", "Content-based", "SVD"])
        n2 = c2.number_input("Jumlah ", 5, 50, 10)
        sc = scores_from_items(M, picks, rts, meth2)
        top = top_n(sc, int(n2), dedup, M)
        st.dataframe(book_table(M, top, sc[top])[["Judul", "Penulis", "Tahun", "Skor"]],
                     )

# ---------------- Buku mirip
with tab5:
    pop_idx = [int(x) for x in np.argsort(-M["cnt"])]
    lab = lambda i: f"{M['binfo'].Title.get(M['inv_i'][i])} — {M['binfo'].Author.get(M['inv_i'][i])}"
    i = st.selectbox("Pilih buku", pop_idx, format_func=lab)
    n3 = st.number_input("Jumlah  ", 5, 30, 10)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mirip berdasarkan rating (Item-kNN cosine)**")
        s = M["S"][i].toarray()[0]
        nz = np.flatnonzero(s)
        if len(nz) == 0:
            st.caption("Tidak ada tetangga.")
        else:
            top = nz[np.argsort(-s[nz])][:int(n3)]
            st.dataframe(book_table(M, top, s[top])[["Judul", "Penulis", "Skor"]],
                         )
    with c2:
        st.markdown("**Mirip berdasarkan konten (TF-IDF)**")
        s = (M["X"][i] @ M["X"].T).toarray()[0]; s[i] = -np.inf
        top = top_n(s, int(n3), dedup, M)
        st.dataframe(book_table(M, top, s[top])[["Judul", "Penulis", "Skor"]],
                     )

# ---------------- Populer
with tab6:
    n4 = st.number_input("Jumlah   ", 5, 100, 20)
    ps = M["pop_score"]
    top = top_n(ps.copy(), int(n4), dedup, M)
    df = book_table(M, top, ps[top]); df["Jumlah rating"] = M["cnt"][top]
    st.caption("Weighted rating (IMDB-style): m = persentil-75 jumlah rating.")
    st.dataframe(df[["Judul", "Penulis", "Jumlah rating", "Skor"]])
