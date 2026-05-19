## 北尾早霧・砂川武貴・山田知明『定量的マクロ経済学と数値計算』日本評論社

### 第5章：格差をどう説明するか ビューリー・モデルによるアプローチ

本リポジトリは、北尾早霧・砂川武貴・山田知明『[定量的マクロ経済学と数値計算](https://www.nippyo.co.jp/shop/book/9287.html)』（日本評論社）の**第5章**に対応するサポートコードを収録しています。

---

### フォルダ構成

```
chapter5/
├── Julia/
│   ├── aiyagari_1994.jl       # Aiyagari (1994) モデルのメイン（均衡計算）
│   ├── aiyagari_vfi1.jl       # VFI 実装（基本版）
│   ├── aiyagari_vfi2.jl       # VFI 実装（改良版）
│   ├── aiyagari_vfi3.jl       # VFI 実装（高速化版）
│   ├── aiyagari_transition.jl  # 移行ダイナミクスの計算
│   └── tauchen.jl             # AR(1) 過程の Tauchen 離散近似
└── MATLAB/
    ├── aiyagari_1994.m        # Aiyagari (1994) モデルのメイン（均衡計算）
    ├── aiyagari_vfi1.m        # VFI 実装（基本版）
    ├── aiyagari_vfi2.m        # VFI 実装（改良版）
    ├── aiyagari_vfi3.m        # VFI 実装（高速化版）
    ├── aiyagari_transition.m  # 移行ダイナミクスの計算
    └── tauchen.m              # AR(1) 過程の Tauchen 離散近似
```

また、Pythonによる再現コードが `Python/Aiyagari_1994.ipynb` に収録されています。

---

### 各言語のコードについて

#### Julia & MATLAB

本章は Bewley–Aiyagari–Huggett–Imrohoro**ğ**lu モデルを解くためのコードのみなので、節ごとのサブフォルダはありません。

各ファイルの役割は以下の通りです。

**`aiyagari_vfi1` / `aiyagari_vfi2` / `aiyagari_vfi3`**：家計の最適化問題（無限期間の資産選択問題）をVFI（価値関数反復法）で解くファイルです。vfi1 が最もシンプルな実装、vfi2 以降は高速化・精度向上を図っています。

**`aiyagari_1994`**：VFI の結果を所与として、資産の定常分布を計算し、資本市場の清算条件（集計資産 = 定常資本）を満たす均衡利子率 $r^*$ を求めるメインファイルです。均衡の外側ループでは二分法または直接探索により利子率を更新します。

**`aiyagari_transition`**：政策変更などのショックに対する移行ダイナミクスを計算します。定常均衡間の移行経路を追跡し、資本ストックの時系列を記録します。

**`tauchen`**：所得（労働生産性）の AR(1) 過程を Tauchen (1986) 法で離散的なマルコフ連鎖に近似します。状態数 $N$ と近似幅 $m$ を引数として遷移確率行列を返します。

---

### コードの構成と計算の流れ

1. **Tauchen 離散化**（`tauchen`）：AR(1) 所得過程 $\log z' = \rho \log z + \varepsilon$ を $N$ 点のマルコフ連鎖に変換
2. **家計問題の VFI**（`aiyagari_vfi*`）：所与の利子率 $r$ のもとで価値関数を反復計算し、政策関数 $a' = g(a, z)$ を求める
3. **定常分布の計算**（`aiyagari_1994`）：遷移確率行列と政策関数から資産の定常分布 $\lambda(a, z)$ を反復計算で導出
4. **均衡の探索**：集計資産 $\int a \, d\lambda = K$ が定常状態の資本需要 $K(r)$ と一致する利子率 $r^*$ を求める

---

### Python について

* RAの小野泰輝さんに作成していただいたPython用再現コードです（`Python/Aiyagari_1994.ipynb`）。もちろん残りうるあらゆる間違いは、すべて著者たちの責任です。

---

### 注意

- MATLABについては北尾早霧が作成しています。
- JuliaについてはRAの鈴木徳馬さんに作成していただきました。もちろん残りうるあらゆる間違いは、すべて著者たちの責任です。
- 定常分布の計算は反復回数が多くなるため、収束に時間がかかる場合があります。Julia では `aiyagari_vfi3.jl` などでキャッシュや高速化の工夫が施されています。
- Juliaではいくつかのパッケージを利用しています（`Optim`、`Interpolations` 等）。もし実行できない場合はREPLで`]`を押したのち、`add パッケージ名`を実行してインストールしてください。

---

### セットアップ

Julia と Jupyter Notebook のインストール・環境設定については、[インストールと環境構築のガイド](https://quant-macro-book.github.io/setup/) を参照してください。
