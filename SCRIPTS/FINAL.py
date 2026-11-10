# -*- coding: utf-8 -*-
"""
Detection de ruptures dans les series pluviometriques -- 14 stations, 1961-2025.
Bibliotheque de tests reprise de l'implementation fournie par l'expert
climatologue (equivalent libre des procedures KHRONOSTAT 1.01, IRD 1998) :
    * Pettitt (1979)
    * Buishand (1982), version parametrique (Monte-Carlo sous H0 gaussienne)
    * Lee & Heghinian (1977), approche bayesienne
    * Segmentation de Hubert (Hubert et al., 1989), programmation dynamique
    * Mann-Kendall (tendance, sert aussi de verification prealable)

Worsley n'est pas utilise dans cette version.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import stats
import openpyxl
import os


# ==========================================================================
# 1. PETTITT (1979)
# ==========================================================================
@dataclass
class ResultatRupture:
    test: str
    n: int
    statistique: float
    p_value: float
    indice_rupture: int | None
    date_rupture: object
    moyenne_avant: float
    moyenne_apres: float
    variation_pct: float
    significatif_5pct: bool


def pettitt(x, dates=None, alpha: float = 0.05) -> ResultatRupture:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x, idx = x[ok], np.where(ok)[0]
    n = x.size
    if n < 5:
        raise ValueError("serie trop courte")

    rangs = stats.rankdata(x)
    U = 2 * np.cumsum(rangs) - np.arange(1, n + 1) * (n + 1)
    k = int(np.argmax(np.abs(U[:-1])))
    K = float(np.abs(U[k]))
    p = float(min(1.0, 2.0 * np.exp(-6.0 * K ** 2 / (n ** 3 + n ** 2))))

    m1, m2 = float(np.mean(x[:k + 1])), float(np.mean(x[k + 1:]))
    return ResultatRupture(
        "Pettitt", n, K, p, int(idx[k]),
        _date(dates, idx[k]), round(m1, 2), round(m2, 2),
        round(100 * (m2 - m1) / m1, 1) if m1 else np.nan, p < alpha)


# ==========================================================================
# 2. BUISHAND (1982) -- version parametrique (comme l'expert)
# ==========================================================================
def buishand(x, dates=None, alpha: float = 0.05, n_mc: int = 5000,
             graine: int = 42) -> dict:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x, idx = x[ok], np.where(ok)[0]
    n = x.size
    if n < 5:
        raise ValueError("serie trop courte")

    def stats_buishand(v):
        s = np.concatenate([[0.0], np.cumsum(v - v.mean())])
        sd = v.std(ddof=1)
        if sd == 0:
            return 0.0, 0.0, s
        ss = s / sd
        U = float(np.sum(ss[1:-1] ** 2) / (len(v) * (len(v) + 1)))
        R = float((ss.max() - ss.min()) / np.sqrt(len(v)))
        return U, R, ss

    U, R, ss = stats_buishand(x)
    k = int(np.argmax(np.abs(ss[1:-1])))

    rng = np.random.default_rng(graine)
    sim = rng.standard_normal((n_mc, n))
    Us = np.empty(n_mc)
    Rs = np.empty(n_mc)
    for i in range(n_mc):
        Us[i], Rs[i], _ = stats_buishand(sim[i])
    pU = float((np.sum(Us >= U) + 1) / (n_mc + 1))
    pR = float((np.sum(Rs >= R) + 1) / (n_mc + 1))

    m1, m2 = float(np.mean(x[:k + 1])), float(np.mean(x[k + 1:]))
    return {
        "test": "Buishand", "n": n, "U": round(U, 4), "p_value_U": round(pU, 4),
        "R_reduit": round(R, 4), "p_value_R": round(pR, 4),
        "seuil_U_5pct": round(float(np.quantile(Us, 0.95)), 4),
        "seuil_R_5pct": round(float(np.quantile(Rs, 0.95)), 4),
        "indice_rupture": int(idx[k]), "date_rupture": _date(dates, idx[k]),
        "moyenne_avant": round(m1, 2), "moyenne_apres": round(m2, 2),
        "variation_pct": round(100 * (m2 - m1) / m1, 1) if m1 else np.nan,
        "significatif_5pct": bool(min(pU, pR) < alpha),
    }


# ==========================================================================
# 3. LEE & HEGHINIAN (1977)
# ==========================================================================
def lee_heghinian(x, dates=None) -> dict:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x, idx = x[ok], np.where(ok)[0]
    n = x.size
    if n < 5:
        raise ValueError("serie trop courte")

    taus = np.arange(1, n)
    cs = np.cumsum(x)
    css = np.cumsum(x ** 2)
    tot2 = css[-1]
    s1 = cs[taus - 1]
    s2 = cs[-1] - s1
    n1 = taus.astype(float)
    n2 = n - n1
    R = tot2 - s1 ** 2 / n1 - s2 ** 2 / n2
    R = np.maximum(R, 1e-300)

    log_f = 0.5 * (np.log(n) - np.log(n1) - np.log(n2)) - (n - 2) / 2.0 * np.log(R)
    log_f -= log_f.max()
    post = np.exp(log_f)
    post /= post.sum()

    k = int(taus[np.argmax(post)]) - 1
    ordre = np.argsort(post)[::-1]
    cum = np.cumsum(post[ordre])
    ens = np.sort(taus[ordre[:np.searchsorted(cum, 0.90) + 1]])

    m1, m2 = float(np.mean(x[:k + 1])), float(np.mean(x[k + 1:]))
    return {
        "test": "Lee-Heghinian", "n": n, "indice_rupture": int(idx[k]),
        "date_rupture": _date(dates, idx[k]),
        "proba_posteriori_mode": round(float(post.max()), 4),
        "intervalle_credibilite_90": (_date(dates, idx[int(ens[0]) - 1]),
                                      _date(dates, idx[int(ens[-1]) - 1])),
        "moyenne_avant": round(m1, 2), "moyenne_apres": round(m2, 2),
        "variation_pct": round(100 * (m2 - m1) / m1, 1) if m1 else np.nan,
    }


# ==========================================================================
# 4. SEGMENTATION DE HUBERT (1989) -- programmation dynamique
# ==========================================================================
def hubert(x, dates=None, alpha: float = 0.01, k_max: int = 10) -> dict:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x, idx = x[ok], np.where(ok)[0]
    n = x.size
    if n < 6:
        raise ValueError("serie trop courte")

    cs = np.concatenate([[0.0], np.cumsum(x)])
    css = np.concatenate([[0.0], np.cumsum(x ** 2)])

    def cout(i, j):
        m = j - i
        s = cs[j] - cs[i]
        return (css[j] - css[i]) - s * s / m

    k_max = int(min(k_max, n // 2))
    INF = np.inf
    D = np.full((k_max + 1, n + 1), INF)
    B = np.zeros((k_max + 1, n + 1), dtype=int)
    for j in range(1, n + 1):
        D[1, j] = cout(0, j)
    for k in range(2, k_max + 1):
        for j in range(k, n + 1):
            meilleur, arg = INF, k - 1
            for i in range(k - 1, j):
                if D[k - 1, i] == INF:
                    continue
                v = D[k - 1, i] + cout(i, j)
                if v < meilleur:
                    meilleur, arg = v, i
            D[k, j], B[k, j] = meilleur, arg

    def bornes(k):
        b, j = [n], n
        for kk in range(k, 1, -1):
            j = B[kk, j]
            b.append(j)
        b.append(0)
        return sorted(set(b))

    retenu, detail = 1, None
    for k in range(2, k_max + 1):
        if not np.isfinite(D[k, n]):
            break
        b = bornes(k)
        seg = [(b[m], b[m + 1]) for m in range(len(b) - 1)]
        if any(j - i < 2 for i, j in seg):
            break
        moy = np.array([x[i:j].mean() for i, j in seg])
        tail = np.array([j - i for i, j in seg], dtype=float)
        sce = D[k, n]
        ddl = n - k
        s2 = sce / ddl
        if s2 <= 0:
            break
        Fc = stats.f.ppf(1 - alpha, k - 1, ddl)
        ok_tous = True
        paires = []
        for m in range(k - 1):
            diff = abs(moy[m] - moy[m + 1])
            crit = np.sqrt((k - 1) * Fc * s2 * (1 / tail[m] + 1 / tail[m + 1]))
            paires.append((m, round(float(diff), 2), round(float(crit), 2), diff > crit))
            if diff <= crit:
                ok_tous = False
        if not ok_tous:
            break
        retenu = k
        detail = (seg, moy, paires, sce)

    if detail is None:
        seg, moy = [(0, n)], np.array([x.mean()])
        paires, sce = [], float(np.sum((x - x.mean()) ** 2))
    else:
        seg, moy, paires, sce = detail

    segments = [{
        "segment": m + 1, "debut": _date(dates, idx[i]), "fin": _date(dates, idx[j - 1]),
        "n_annees": int(j - i), "moyenne": round(float(mu), 2),
        "ecart_type": round(float(np.std(x[i:j], ddof=1)) if j - i > 1 else 0.0, 2),
    } for m, ((i, j), mu) in enumerate(zip(seg, moy))]

    return {
        "test": "Hubert (segmentation)", "n": n, "alpha": alpha,
        "ordre_retenu": int(retenu), "n_ruptures": int(retenu - 1),
        "dates_ruptures": [_date(dates, idx[j - 1]) for (i, j) in seg[:-1]],
        "segments": segments, "sce": round(float(sce), 2), "scheffe_contigus": paires,
    }


# ==========================================================================
# 5. FONCTIONS DE VERIFICATION DES HYPOTHESES PREALABLES
#    (independance, tendance, homoscedasticite, normalite)
# ==========================================================================
def mann_kendall(x, alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    s = int(sum(np.sign(x[j] - x[i]).sum() for i, j in
                [(i, slice(i + 1, n)) for i in range(n - 1)]))
    _, comptes = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5)
           - np.sum(comptes * (comptes - 1) * (2 * comptes + 5))) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    pentes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return {"test": "Mann-Kendall", "n": n, "S": s, "Z": round(float(z), 3),
            "p_value": round(p, 4), "pente_sen": round(float(np.median(pentes)), 3),
            "tendance": ("hausse" if z > 0 else "baisse" if z < 0 else "aucune")
                        if p < alpha else "non significative",
            "significatif_5pct": p < alpha}


def autocorrelation_lag1(x):
    """Verification prealable complementaire (independance des observations)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xbar = np.mean(x)
    num = np.sum((x[:-1] - xbar) * (x[1:] - xbar))
    den = np.sum((x - xbar) ** 2)
    r1 = num / den
    seuil = 1.96 / np.sqrt(n)
    return r1, seuil, abs(r1) > seuil


def stabilite_variance(x, alpha=0.05):
    """Verification prealable : HOMOSCEDASTICITE. La serie est coupee en son
    milieu (independamment de toute date de rupture, puisque celle-ci n'est
    pas encore connue a ce stade), et un test de Levene compare les
    variances des deux moities."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    milieu = n // 2
    moitie1, moitie2 = x[:milieu], x[milieu:]
    stat_levene, p_levene = stats.levene(moitie1, moitie2)
    var1, var2 = np.var(moitie1, ddof=1), np.var(moitie2, ddof=1)
    return {
        "variance_moitie1": round(float(var1), 1),
        "variance_moitie2": round(float(var2), 1),
        "levene_stat": round(float(stat_levene), 3),
        "levene_p": round(float(p_levene), 4),
        "variance_instable": p_levene < alpha,
    }


def normalite_shapiro(x, alpha=0.05):
    """Verification prealable : NORMALITE des observations, testee sur la
    serie complete par le test de Shapiro-Wilk."""
    x = np.asarray(x, dtype=float)
    stat_sw, p_sw = stats.shapiro(x)
    return {
        "shapiro_stat": round(float(stat_sw), 4),
        "shapiro_p": round(float(p_sw), 4),
        "normalite_rejetee": p_sw < alpha,
    }


def _date(dates, i):
    if dates is None:
        return int(i)
    v = pd.Series(dates).iloc[int(i)]
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y")
    if isinstance(v, (np.integer, np.floating)):
        return int(v)
    return v


def analyser(serie: pd.Series, alpha: float = 0.05, alpha_hubert: float = 0.01) -> dict:
    s = serie.dropna()
    d = s.index
    return {
        "pettitt": asdict(pettitt(s.values, d, alpha)),
        "buishand": buishand(s.values, d, alpha),
        "lee_heghinian": lee_heghinian(s.values, d),
        "hubert": hubert(s.values, d, alpha_hubert),
        "mann_kendall": mann_kendall(s.values, alpha),
    }


# ==========================================================================
# 6. CORRECTION POUR TESTS MULTIPLES (Benjamini-Hochberg / FDR)
#    Appliquee aux p-values des tests de rupture (Pettitt, Buishand)
# ==========================================================================

def bh_fdr(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    ordre = np.argsort(pvals)
    p_triees = pvals[ordre]
    p_adj_triees = p_triees * n / (np.arange(1, n + 1))
    p_adj_triees = np.minimum.accumulate(p_adj_triees[::-1])[::-1]
    p_adj_triees = np.clip(p_adj_triees, 0, 1)
    p_adj = np.empty(n)
    p_adj[ordre] = p_adj_triees
    return p_adj, p_adj < alpha


# ==========================================================================
# 0. CHARGEMENT DES DONNEES (14 stations, une feuille par station)
# ==========================================================================

FICHIER = r"C:\Users\bmd\Desktop\Stage_Immersion_ISEP2\METHODOLOGIE\Stage_METHODOLOGIE\Cumuls_pluvio_14stations_1961_2025.xlsx"
ANNEE_DEBUT, ANNEE_FIN = 1961, 2025
NB_ANNEES_ATTENDU = ANNEE_FIN - ANNEE_DEBUT + 1

wb = openpyxl.load_workbook(FICHIER, data_only=True)
stations_data = {}
for ws in wb.worksheets:
    rows = list(ws.iter_rows(values_only=True))
    d = {int(r[0]): r[1] for r in rows if r[0] is not None and r[1] is not None}
    annees = sorted(y for y in d if ANNEE_DEBUT <= y <= ANNEE_FIN)
    valeurs = np.array([d[y] for y in annees], dtype=float)
    stations_data[ws.title] = (np.array(annees), valeurs)
    assert len(annees) == NB_ANNEES_ATTENDU, f"{ws.title} : {len(annees)} annees (attendu {NB_ANNEES_ATTENDU})"

print(f"{len(stations_data)} stations chargees, periode {ANNEE_DEBUT}-{ANNEE_FIN}.")


# ==========================================================================
# 7. APPLICATION DES VERIFICATIONS PREALABLES SUR LES 14 STATIONS
#    (independance, tendance, homoscedasticite, normalite)
# ==========================================================================

resultats_verif = []
for station, (annees, x) in stations_data.items():
    # 1. Independance des observations
    r1, seuil_r1, autocorr_sig = autocorrelation_lag1(x)
    # 2. Absence de tendance monotone
    mk = mann_kendall(x)
    # 3. Homoscedasticite
    var_check = stabilite_variance(x)
    # 4. Normalite des observations
    norm_check = normalite_shapiro(x)

    resultats_verif.append({
        "station": station,
        # -- 1. Independance --
        "r1": round(r1, 3), "seuil_r1_5pct": round(seuil_r1, 3),
        "autocorrelation_significative": autocorr_sig,
        # -- 2. Tendance --
        "mk_Z": mk["Z"], "mk_p": mk["p_value"], "tendance_significative": mk["significatif_5pct"],
        "mk_pente_sen": mk["pente_sen"],
        # -- 3. Homoscedasticite --
        "variance_moitie1": var_check["variance_moitie1"],
        "variance_moitie2": var_check["variance_moitie2"],
        "levene_p": var_check["levene_p"],
        "variance_instable": var_check["variance_instable"],
        # -- 4. Normalite --
        "shapiro_stat": norm_check["shapiro_stat"],
        "shapiro_p": norm_check["shapiro_p"],
        "normalite_rejetee": norm_check["normalite_rejetee"],
        # -- Synthese des 4 hypotheses --
        "hypotheses_respectees": (not autocorr_sig) and (not mk["significatif_5pct"])
                                  and (not var_check["variance_instable"])
                                  and (not norm_check["normalite_rejetee"]),
    })
df_verif = pd.DataFrame(resultats_verif).sort_values("station")
print("\n=== VERIFICATION DES HYPOTHESES PREALABLES (independance, tendance, homoscedasticite, normalite) ===")
print(df_verif.to_string(index=False))

nb_ok = df_verif["hypotheses_respectees"].sum()
print(f"\n{nb_ok} / {len(df_verif)} stations respectent les quatre hypotheses.")
viol = df_verif[~df_verif["hypotheses_respectees"]]
if len(viol) > 0:
    print("\nStations en anomalie :")
    for _, r in viol.iterrows():
        pb = []
        if r["autocorrelation_significative"]:
            pb.append(f"independance non verifiee (r1={r['r1']})")
        if r["tendance_significative"]:
            pb.append(f"tendance significative (Z={r['mk_Z']}, p={r['mk_p']})")
        if r["variance_instable"]:
            pb.append(f"homoscedasticite non verifiee (Levene p={r['levene_p']})")
        if r["normalite_rejetee"]:
            pb.append(f"normalite non verifiee (Shapiro p={r['shapiro_p']})")
        print(f"  - {r['station']} : {', '.join(pb)}")


# ==========================================================================
# 8. APPLICATION DES 4 PROCEDURES SUR LES 14 STATIONS
# ==========================================================================

lignes = []
segments_hubert_tout = []
for station, (annees, x) in stations_data.items():
    serie = pd.Series(x, index=annees)
    r = analyser(serie)

    hubert_dates = r["hubert"]["dates_ruptures"]
    hubert_principale = hubert_dates[0] if hubert_dates else None

    lignes.append({
        "station": station,
        "pettitt_annee": r["pettitt"]["date_rupture"],
        "pettitt_p": r["pettitt"]["p_value"],
        "pettitt_sig": r["pettitt"]["significatif_5pct"],
        "buishand_annee": r["buishand"]["date_rupture"],
        "buishand_p": round(min(r["buishand"]["p_value_U"], r["buishand"]["p_value_R"]), 4),
        "buishand_sig": r["buishand"]["significatif_5pct"],
        "lee_heghinian_annee": r["lee_heghinian"]["date_rupture"],
        "lee_heghinian_proba": r["lee_heghinian"]["proba_posteriori_mode"],
        "hubert_n_ruptures": r["hubert"]["n_ruptures"],
        "hubert_dates": hubert_dates,
        "hubert_annee_principale": hubert_principale,
        "mk_Z": r["mann_kendall"]["Z"],
        "mk_p": r["mann_kendall"]["p_value"],
        "mk_pente_sen": r["mann_kendall"]["pente_sen"],
    })
    for sg in r["hubert"]["segments"]:
        segments_hubert_tout.append({"station": station, **sg})

df_final = pd.DataFrame(lignes)

# Correction FDR (Benjamini-Hochberg) sur Pettitt et Buishand
for test in ["pettitt", "buishand"]:
    p_adj, sig_adj = bh_fdr(df_final[f"{test}_p"].values)
    df_final[f"{test}_p_fdr"] = np.round(p_adj, 4)
    df_final[f"{test}_signif_fdr"] = sig_adj

print("\n=== TABLEAU FINAL (4 procedures) ===")
print(df_final.to_string(index=False))


# ==========================================================================
# 9. REGLE DE VALIDATION PAR MAJORITE (>= 3 procedures sur 4 convergentes)
# ==========================================================================

def annees_par_test(row):
    return [row["pettitt_annee"], row["buishand_annee"],
            row["lee_heghinian_annee"], row["hubert_annee_principale"]]


def rupture_majoritaire(row, tolerance=3):
    """Regroupe les dates a 'tolerance' annees pres. En cas d'egalite de
    score entre plusieurs dates candidates, on retient celle qui est
    EXACTEMENT la plus frequente parmi les tests (le veritable consensus),
    plutot que la premiere rencontree dans l'ordre des tests."""
    from collections import Counter
    dates = [d for d in annees_par_test(row) if d is not None]
    if not dates:
        return None, 0
    freq = Counter(dates)
    meilleur_score, candidats = 0, []
    for d in set(dates):
        score = sum(1 for d2 in dates if abs(d2 - d) <= tolerance)
        if score > meilleur_score:
            meilleur_score, candidats = score, [d]
        elif score == meilleur_score:
            candidats.append(d)
    meilleure_date = max(candidats, key=lambda d: freq[d])
    return meilleure_date, meilleur_score


df_final["date_majoritaire"] = df_final.apply(lambda r: rupture_majoritaire(r)[0], axis=1)
df_final["nb_tests_convergents"] = df_final.apply(lambda r: rupture_majoritaire(r)[1], axis=1)
df_final["rupture_averee"] = df_final["nb_tests_convergents"] >= 3

print("\n=== VALIDATION PAR MAJORITE (>= 3 procedures sur 4) ===")
print(df_final[["station", "date_majoritaire", "nb_tests_convergents", "rupture_averee"]].to_string(index=False))


# ==========================================================================
# 10. EXPORT EXCEL
# ==========================================================================

DOSSIER_SORTIE = os.path.dirname(FICHIER)
try:
    with pd.ExcelWriter(os.path.join(DOSSIER_SORTIE, "resultats_4tests_ruptures.xlsx")) as writer:
        df_final.to_excel(writer, sheet_name="resultats", index=False)
        df_verif.to_excel(writer, sheet_name="verif_hypotheses", index=False)
        pd.DataFrame(segments_hubert_tout).to_excel(writer, sheet_name="segments_hubert", index=False)
    print(f"\nFichier exporte : resultats_4tests_ruptures.xlsx")
except PermissionError:
    import datetime
    horodatage = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin = os.path.join(DOSSIER_SORTIE, f"resultats_4tests_ruptures_{horodatage}.xlsx")
    with pd.ExcelWriter(chemin) as writer:
        df_final.to_excel(writer, sheet_name="resultats", index=False)
        df_verif.to_excel(writer, sheet_name="verif_hypotheses", index=False)
        pd.DataFrame(segments_hubert_tout).to_excel(writer, sheet_name="segments_hubert", index=False)
    print(f"[!] Fichier verrouille, exporte sous : {chemin}")


# ==========================================================================
# 11. GRAPHIQUES
# ==========================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# --- Grille des 14 series avec rupture ---
fig, axes = plt.subplots(7, 2, figsize=(14, 24), sharex=True)
axes = axes.flatten()
for i, station in enumerate(df_final["station"]):
    annees, x = stations_data[station]
    ax = axes[i]
    ax.plot(annees, x, color="steelblue", linewidth=1.2)
    rupture = df_final.loc[df_final["station"] == station, "date_majoritaire"].values[0]
    rupture_averee = df_final.loc[df_final["station"] == station, "rupture_averee"].values[0]
    if rupture_averee:
        rupture = int(rupture)
        ax.axvline(rupture, color="firebrick", linestyle="--", linewidth=1.5)
        avant, apres = x[annees <= rupture], x[annees > rupture]
        ax.hlines(avant.mean(), annees.min(), rupture, color="darkgreen", linewidth=2)
        ax.hlines(apres.mean(), rupture, annees.max(), color="darkorange", linewidth=2)
        ax.set_title(f"{station}  (rupture {rupture})", fontsize=11)
    else:
        ax.set_title(f"{station}  (pas de rupture avérée)", fontsize=11, color="gray")
    ax.set_ylabel("mm")
fig.suptitle("Cumuls pluviométriques annuels et rupture détectée (1961-2025)", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig("graph1_series_ruptures.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Heatmap de significativite (Pettitt, Buishand) ---
sig_matrix = df_final.set_index("station")[["pettitt_sig", "buishand_sig"]].astype(int)
sig_matrix.columns = ["Pettitt", "Buishand"]
fig, ax = plt.subplots(figsize=(5, 8))
sns.heatmap(sig_matrix, cmap=["#f0f0f0", "#2e7d32"], cbar=False, linewidths=1, linecolor="white",
            annot=sig_matrix.replace({1: "Sig.", 0: ""}), fmt="", ax=ax)
ax.set_title("Significativité des tests (p < 0.05)")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig("graph2_heatmap_signif.png", dpi=150)
plt.close(fig)

# --- Convergence des 4 procedures (barres) ---
df_sorted = df_final.sort_values("nb_tests_convergents")
colors = ["firebrick" if v < 3 else "steelblue" for v in df_sorted["nb_tests_convergents"]]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(df_sorted["station"], df_sorted["nb_tests_convergents"], color=colors)
ax.axvline(3, color="black", linestyle="--", linewidth=1, label="Seuil de majorité (3/4)")
ax.set_xlabel("Nombre de procédures convergentes (sur 4)")
ax.set_title("Robustesse de la rupture détectée par station")
ax.legend()
fig.tight_layout()
fig.savefig("graph3_convergence.png", dpi=150)
plt.close(fig)

print("\n3 graphiques generes : graph1_series_ruptures.png, graph2_heatmap_signif.png, graph3_convergence.png")

# ==========================================================================
# 11 bis. FIGURES INDIVIDUELLES (une par station)
# ==========================================================================
DOSSIER_FIG_IND = os.path.join(DOSSIER_SORTIE, "figures_individuelles_14stations")
os.makedirs(DOSSIER_FIG_IND, exist_ok=True)

for station in df_final["station"]:
    annees, x = stations_data[station]
    ligne = df_final.loc[df_final["station"] == station].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(annees, x, color="steelblue", linewidth=1.0)

    if ligne["rupture_averee"]:
        rupture = int(ligne["date_majoritaire"])
        avant, apres = x[annees <= rupture], x[annees > rupture]
        ax.axvline(rupture, color="darkred", linestyle="--", linewidth=1.2)
        ax.hlines(avant.mean(), annees.min(), rupture, color="darkgreen", linewidth=2)
        ax.hlines(apres.mean(), rupture, annees.max(), color="darkorange", linewidth=2)

    ax.set_xlabel("Année")
    ax.set_ylabel("Cumul annuel (mm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER_FIG_IND, f"{station}.png"), dpi=200)
    plt.close(fig)

print(f"14 figures enregistrées dans : {DOSSIER_FIG_IND}")

# ==========================================================================
# 12. COMPARAISON DES MOYENNES AVANT / APRES RUPTURE (test t / Welch)
#     S'appuie sur la date de rupture retenue (df_final["date_majoritaire"])
# ==========================================================================
def comparaison_moyennes(x, annees, date_rupture, alpha: float = 0.05) -> dict:
    """Compare les moyennes avant/apres une date de rupture donnee.
    Utilise le test de Levene pour verifier l'homoscedasticite, puis bascule
    automatiquement sur le test de Welch si l'hypothese n'est pas verifiee."""
    x = np.asarray(x, dtype=float)
    annees = np.asarray(annees)
    avant = x[annees <= date_rupture]
    apres = x[annees > date_rupture]

    if len(avant) < 2 or len(apres) < 2:
        return {"erreur": "sous-periode trop courte pour comparer les moyennes"}

    # Verification prealable : homoscedasticite (Levene)
    stat_levene, p_levene = stats.levene(avant, apres)
    variances_egales = p_levene >= alpha

    # Verification prealable : normalite de chaque sous-periode (Shapiro)
    _, p_norm_avant = stats.shapiro(avant)
    _, p_norm_apres = stats.shapiro(apres)

    if variances_egales:
        t_stat, p_val = stats.ttest_ind(avant, apres, equal_var=True)
        test_utilise = "Student (variances egales)"
    else:
        t_stat, p_val = stats.ttest_ind(avant, apres, equal_var=False)
        test_utilise = "Welch (variances inegales)"

    m1, m2 = float(np.mean(avant)), float(np.mean(apres))
    return {
        "date_rupture": date_rupture,
        "n_avant": len(avant), "n_apres": len(apres),
        "moyenne_avant": round(m1, 2), "moyenne_apres": round(m2, 2),
        "variation_pct": round(100 * (m2 - m1) / m1, 1) if m1 else np.nan,
        "levene_p": round(float(p_levene), 4), "variances_egales": bool(variances_egales),
        "shapiro_p_avant": round(float(p_norm_avant), 4),
        "shapiro_p_apres": round(float(p_norm_apres), 4),
        "test_utilise": test_utilise,
        "t_stat": round(float(t_stat), 3), "p_value": round(float(p_val), 4),
        "difference_significative_5pct": bool(p_val < alpha),
    }


# Application sur les 14 stations dont la rupture est jugee averee
resultats_comparaison = []
for _, row in df_final.iterrows():
    station = row["station"]
    annees, x = stations_data[station]
    if row["rupture_averee"] and row["date_majoritaire"] is not None:
        res = comparaison_moyennes(x, annees, int(row["date_majoritaire"]))
        res["station"] = station
        resultats_comparaison.append(res)

df_comparaison = pd.DataFrame(resultats_comparaison)
cols_ordre = ["station", "date_rupture", "n_avant", "n_apres", "moyenne_avant",
              "moyenne_apres", "variation_pct", "variances_egales", "test_utilise",
              "t_stat", "p_value", "difference_significative_5pct"]
df_comparaison = df_comparaison[cols_ordre]
print("\n=== COMPARAISON DES MOYENNES AVANT/APRES RUPTURE ===")
print(df_comparaison.to_string(index=False))


# ==========================================================================
# 13. ANALYSE EN COMPOSANTES PRINCIPALES (ACP)
#     Matrice : 65 annees (lignes) x 14 stations (colonnes)
# ==========================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def construire_matrice_acp(stations_data: dict) -> pd.DataFrame:
    """Construit la matrice annees x stations a partir du dict stations_data."""
    # toutes les stations partagent les memes annees (verifie au chargement)
    premiere_station = next(iter(stations_data))
    annees_ref = stations_data[premiere_station][0]
    df = pd.DataFrame(
        {station: valeurs for station, (annees, valeurs) in stations_data.items()},
        index=annees_ref
    )
    df.index.name = "annee"
    return df


def analyser_acp(df_annees_stations: pd.DataFrame, seuil_kaiser: float = 1.0) -> dict:
    """Centre-reduit les donnees, calcule l'ACP, et retient les axes selon
    le critere de Kaiser (valeur propre > 1)."""
    Z = StandardScaler().fit_transform(df_annees_stations.values)

    pca = PCA()
    F = pca.fit_transform(Z)  # coordonnees factorielles des individus (annees)
    valeurs_propres = pca.explained_variance_
    part_expliquee = pca.explained_variance_ratio_ * 100
    part_cumulee = np.cumsum(part_expliquee)

    # Critere de Kaiser : on ne garde que les axes de valeur propre > seuil
    nb_axes_kaiser = int(np.sum(valeurs_propres > seuil_kaiser))
    nb_axes_kaiser = max(nb_axes_kaiser, 2)  # au moins 2 axes pour un plan factoriel

    # Coordonnees des variables (stations) sur les axes retenus : cercle des correlations
    # corr(variable_j, axe_k) = loadings * sqrt(valeur propre_k)
    loadings = pca.components_.T * np.sqrt(valeurs_propres)
    df_loadings = pd.DataFrame(
        loadings[:, :nb_axes_kaiser],
        index=df_annees_stations.columns,
        columns=[f"Axe{k+1}" for k in range(nb_axes_kaiser)]
    )

    df_coord_annees = pd.DataFrame(
        F[:, :nb_axes_kaiser],
        index=df_annees_stations.index,
        columns=[f"Axe{k+1}" for k in range(nb_axes_kaiser)]
    )

    df_valeurs_propres = pd.DataFrame({
        "axe": [f"Axe{k+1}" for k in range(len(valeurs_propres))],
        "valeur_propre": np.round(valeurs_propres, 3),
        "part_expliquee_pct": np.round(part_expliquee, 2),
        "part_cumulee_pct": np.round(part_cumulee, 2),
        "retenu_kaiser": valeurs_propres > seuil_kaiser,
    })

    return {
        "nb_axes_retenus": nb_axes_kaiser,
        "valeurs_propres": df_valeurs_propres,
        "coordonnees_annees": df_coord_annees,
        "loadings_stations": df_loadings,
        "pca_objet": pca,
    }


df_acp_brut = construire_matrice_acp(stations_data)
resultat_acp = analyser_acp(df_acp_brut)

print("\n=== ACP : VALEURS PROPRES ET PART DE VARIANCE EXPLIQUEE ===")
print(resultat_acp["valeurs_propres"].to_string(index=False))
print(f"\nNombre d'axes retenus (critere de Kaiser) : {resultat_acp['nb_axes_retenus']}")

print("\n=== ACP : CONTRIBUTION DES STATIONS AUX AXES RETENUS (loadings) ===")
print(resultat_acp["loadings_stations"].round(3).to_string())

print("\n=== ACP : COORDONNEES DES ANNEES SUR LES AXES RETENUS (extrait) ===")
print(resultat_acp["coordonnees_annees"].head(10).round(3).to_string())


# ==========================================================================
# 14. CLASSIFICATION DES ANNEES (CAH, critere de Ward)
#     Appliquee aux coordonnees factorielles des axes retenus par Kaiser
# ==========================================================================
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

def classifier_annees(df_coord: pd.DataFrame, nb_classes: int = 3) -> dict:
    """Classification ascendante hierarchique (Ward, distance euclidienne)
    sur les coordonnees factorielles issues de l'ACP."""
    Z_link = linkage(df_coord.values, method="ward", metric="euclidean")

    # Saut d'inertie entre chaque fusion (pour justifier/verifier nb_classes)
    sauts_inertie = np.diff(Z_link[:, 2])
    # indice du plus grand saut, en partant de la fin (fusions les plus hautes)
    idx_saut_max = int(np.argmax(sauts_inertie[-10:])) + (len(sauts_inertie) - 10)
    nb_classes_suggere = len(df_coord) - (idx_saut_max + 1)

    classes = fcluster(Z_link, t=nb_classes, criterion="maxclust")
    df_classes = pd.DataFrame({
        "annee": df_coord.index,
        "classe": classes,
    })

    # Caracterisation des classes a partir de la moyenne des cumuls pluvio
    # (necessite la matrice brute annees x stations, passee en argument externe)
    return {
        "linkage": Z_link,
        "df_classes": df_classes,
        "nb_classes_utilise": nb_classes,
        "nb_classes_suggere_par_inertie": nb_classes_suggere,
    }


resultat_cah = classifier_annees(resultat_acp["coordonnees_annees"], nb_classes=3)
df_classes_annees = resultat_cah["df_classes"]

print(f"\n=== CLASSIFICATION (CAH, Ward) : {resultat_cah['nb_classes_utilise']} classes retenues ===")
print(f"(nombre de classes suggere par le plus grand saut d'inertie : "
      f"{resultat_cah['nb_classes_suggere_par_inertie']})")

# Caracterisation des classes : moyenne pluviometrique nationale par classe
df_moy_nationale = df_acp_brut.mean(axis=1).rename("moyenne_nationale")
df_classes_annees = df_classes_annees.merge(
    df_moy_nationale, left_on="annee", right_index=True
)
synthese_classes = df_classes_annees.groupby("classe")["moyenne_nationale"].agg(
    ["mean", "min", "max", "count"]
).round(1)
synthese_classes.columns = ["moyenne", "min", "max", "nb_annees"]

# Etiquetage automatique des classes selon leur moyenne (sec / normal / humide)
ordre_classes = synthese_classes["moyenne"].sort_values().index
etiquettes = {ordre_classes[0]: "Sec", ordre_classes[1]: "Normal",
              ordre_classes[-1]: "Humide"}
synthese_classes["regime"] = synthese_classes.index.map(etiquettes)

print("\n=== SYNTHESE DES CLASSES (regimes pluviometriques) ===")
print(synthese_classes.to_string())

print("\n=== DETAIL DES ANNEES PAR CLASSE (extrait) ===")
df_classes_annees["regime"] = df_classes_annees["classe"].map(etiquettes)
print(df_classes_annees.sort_values(["classe", "annee"]).head(15).to_string(index=False))


# ==========================================================================
# 15. GRAPHIQUES ACP + CLASSIFICATION
# ==========================================================================
# --- Eboulis des valeurs propres (scree plot) ---
fig, ax = plt.subplots(figsize=(7, 5))
vp = resultat_acp["valeurs_propres"]
ax.bar(vp["axe"], vp["valeur_propre"], color="steelblue")
ax.axhline(1.0, color="firebrick", linestyle="--", label="Seuil de Kaiser (=1)")
ax.set_ylabel("Valeur propre")
ax.set_title("Éboulis des valeurs propres (scree plot)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig("graph4_eboulis_acp.png", dpi=150)
plt.close(fig)

# --- Cercle des correlations (axes 1 et 2) ---
fig, ax = plt.subplots(figsize=(7, 7))
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=0.8)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
for station in resultat_acp["loadings_stations"].index:
    x_l = resultat_acp["loadings_stations"].loc[station, "Axe1"]
    y_l = resultat_acp["loadings_stations"].loc[station, "Axe2"]
    ax.arrow(0, 0, x_l, y_l, head_width=0.02, color="darkorange")
    ax.text(x_l * 1.08, y_l * 1.08, station, fontsize=8)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(f"Axe 1 ({vp.loc[0, 'part_expliquee_pct']}%)")
ax.set_ylabel(f"Axe 2 ({vp.loc[1, 'part_expliquee_pct']}%)")
ax.set_title("Cercle des corrélations (stations)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("graph5_cercle_correlations.png", dpi=150)
plt.close(fig)

# --- Projection des annees, coloree par classe ---
fig, ax = plt.subplots(figsize=(8, 7))
couleurs = {"Sec": "firebrick", "Normal": "gray", "Humide": "steelblue"}
for regime, sous_df in df_classes_annees.merge(
        resultat_acp["coordonnees_annees"], left_on="annee", right_index=True
    ).groupby("regime"):
    ax.scatter(sous_df["Axe1"], sous_df["Axe2"], label=regime,
               color=couleurs.get(regime, "black"), alpha=0.7)
    for _, r in sous_df.iterrows():
        ax.annotate(str(int(r["annee"])), (r["Axe1"], r["Axe2"]), fontsize=6)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_xlabel(f"Axe 1 ({vp.loc[0, 'part_expliquee_pct']}%)")
ax.set_ylabel(f"Axe 2 ({vp.loc[1, 'part_expliquee_pct']}%)")
ax.set_title("Projection des années selon les régimes pluviométriques")
ax.legend()
fig.tight_layout()
fig.savefig("graph6_projection_annees.png", dpi=150)
plt.close(fig)

# --- Dendrogramme ---
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(resultat_cah["linkage"], labels=df_acp_brut.index.astype(str).tolist(),
           ax=ax, leaf_rotation=90, leaf_font_size=7, color_threshold=None)
ax.set_title("Dendrogramme (CAH, critère de Ward)")
ax.set_ylabel("Inertie")
fig.tight_layout()
fig.savefig("graph7_dendrogramme.png", dpi=150)
plt.close(fig)

print("\n4 graphiques generes : graph4_eboulis_acp.png, graph5_cercle_correlations.png, "
      "graph6_projection_annees.png, graph7_dendrogramme.png")


# ==========================================================================
# 16. EXPORT EXCEL COMPLEMENTAIRE
# ==========================================================================
chemin_export2 = os.path.join(DOSSIER_SORTIE, "resultats_acp_classification.xlsx")
try:
    with pd.ExcelWriter(chemin_export2) as writer:
        df_comparaison.to_excel(writer, sheet_name="comparaison_moyennes", index=False)
        resultat_acp["valeurs_propres"].to_excel(writer, sheet_name="acp_valeurs_propres", index=False)
        resultat_acp["loadings_stations"].to_excel(writer, sheet_name="acp_loadings")
        resultat_acp["coordonnees_annees"].to_excel(writer, sheet_name="acp_coord_annees")
        df_classes_annees.to_excel(writer, sheet_name="classification_annees", index=False)
        synthese_classes.to_excel(writer, sheet_name="synthese_classes")
    print(f"\nFichier exporte : {os.path.basename(chemin_export2)}")
except PermissionError:
    import datetime
    horodatage = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_export2 = os.path.join(DOSSIER_SORTIE, f"resultats_acp_classification_{horodatage}.xlsx")
    with pd.ExcelWriter(chemin_export2) as writer:
        df_comparaison.to_excel(writer, sheet_name="comparaison_moyennes", index=False)
        resultat_acp["valeurs_propres"].to_excel(writer, sheet_name="acp_valeurs_propres", index=False)
        resultat_acp["loadings_stations"].to_excel(writer, sheet_name="acp_loadings")
        resultat_acp["coordonnees_annees"].to_excel(writer, sheet_name="acp_coord_annees")
        df_classes_annees.to_excel(writer, sheet_name="classification_annees", index=False)
        synthese_classes.to_excel(writer, sheet_name="synthese_classes")
    print(f"[!] Fichier verrouille, exporte sous : {os.path.basename(chemin_export2)}")


# ==========================================================================
# AJOUT (apres la version du 14 octobre) : analyse de sensibilite a la
# tolerance de concordance des dates (0 a 5 ans). A placer apres le calcul
# de df_final et la fonction rupture_majoritaire(row, tolerance).
# ==========================================================================
sens = []
for tol in range(0, 6):
    res = df_final.apply(lambda r: rupture_majoritaire(r, tolerance=tol), axis=1)
    nb = res.apply(lambda t: t[1])
    dm = res.apply(lambda t: t[0])
    sens.append({"tolerance": tol,
                 "nb_regions_averees": int((nb >= 3).sum()),
                 "regions": ", ".join(df_final.loc[nb >= 3, "station"])})
df_sens = pd.DataFrame(sens)
print(df_sens.to_string(index=False))
df_sens.to_excel("sensibilite_tolerance.xlsx", index=False)
