# CVAE におけるクラスタリングと改善案まとめ

## 現状の観察
- Vanilla VAE: latent に犬/猫情報が残りやすく、分類精度 ~80%
- CVAE: decoder/encoder にラベル壁紙を使用 → latent がクラス情報を持たず ~50%

## なぜ CVAE でクラスタが消えた？
### 原因
- **ラベル壁紙 (one-hot を空間方向に複製)** が decoder にも与えられたことで、
  モデルが **latent にクラス情報を入れる必要がなくなる**
- KL 項が latent を押しつぶし、さらにクラス情報を latent から排除

### 結果
> latent は「背景やノイズの説明係」に追いやられ、分類不可に

## CVAE の設計指針
|設計スタイル|目的|ラベル壁紙|latent の性質|用途|
|---|---|---|---|---|
表現学習・解釈性重視|latent に意味を持たせたい|❌ 非推奨|クラス特徴を保持|クラスタリング, disentanglement|
生成性能重視|条件を強く反映したい|✅ あり得る|latent にクラスが入らない|確実な条件付き生成|

## 改善策まとめ

### 必須 (高優先度)
- ✅ **Decoder からラベル壁紙を外す**
- ✅ **KL アニーリング or β < 1** で latent collapse 防止

### 有効な追加手法
- ⭐ latent に **classification loss を追加**
```python
logits = clf(z)
loss += λ * CE(logits, y)  # λ=0.1〜0.5
```
- latent dim 増加
- one-hot ではなく **trainable label embedding**
- p(z|y) の導入 (class-specific prior)

## 期待される結果
- latent が犬/猫を自然に分け始める
- t-SNE/UMAP でクラスタ形成
- ロジスティック分類精度向上

## 実験順序のおすすめ
1. Decoder からラベル壁紙除去
2. KL annealing
3. latent classifier head 追加
4. 必要なら p(z|y) や InfoVAE/MMD-VAE 方向へ

## 要点の一句
> **「生成より表現」なら、ラベルは“塗らずに渡す”。**

CVAE は「条件を与える」だけでなく  
**“latent にどう責任を持たせるか”を調整できる学習器**。  
今回の改善で、latent が再び語り始めるはず。

