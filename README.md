# ahc030

油田の位置を推定するやつ

seed:0 / cost:1.13
![vis](/docs/vis.gif)

## keywords
- 正規分布（ガウス分布）
  - 累積分布関数
- ベイズの定理
- 尤度
- エントロピー
- 相互情報量
- 山登り，焼きなまし
- ノルム最小化
  - L1ノルム
  - L2ノルム
  - 線形計画問題に定式化
  - 凸最適化（OR学会の連続最適化セッションで言ってたやつ）
- 最小二乗法
- 最尤推定
- MCMC

## note
- インタラクティブ問題：Rustなら `proconio::input_interactive!` 使うと楽
- `std::cmp::Ord is not implemented for float`：めんどい
- 対数尤度：結局どうやって実装したらいいのかよく分からん．`log`とって積を和にするだけ？
- ローカルテスタ：そろそろ整えたい
  - スコア計算：`tools/src/bin/vis.rs`でできる．自分で書かなくて良い．インタラクティブ問題なら，`tester` でも計算してくれる
  - パラメータチューニング：optuna使うときは `.py` よりも `.ipynb` の方が良さそう
  - optunaなどで，パラメータの渡し方：`std::env::args()` で良いの？
  - `IF LOCAL` 判定：<https://zenn.dev/tipstar0125/articles/245bceec86e40a>

## refs
- 問題：<https://atcoder.jp/contests/ahc030/tasks/ahc030_a>
- 公式解説／ユーザー解説：<https://atcoder.jp/contests/ahc030/editorial>
  - ベイズ，相互情報量，焼きなまし（wataさん）：<https://img.atcoder.jp/ahc030/ahc030.pdf>
  - ベイズ推定（あぷりしあさん）：<https://qiita.com/aplysia/items/c3f2111110ac5043710a>
  - エントロピーとか（てりーさん）：<https://www.terry-u16.net/entry/ahc030>
  - ノルム最小化（hari64さん）：<https://qiita.com/hari64/items/a7793a7071b4015ef92c>
  - （eijirouさん）<https://eijirou-kyopro.hatenablog.com/entry/2024/02/22/152604>
  - （よすぽさん）<https://yosupo.hatenablog.com/entry/2024/02/21/044022>
- 参考になる提出コード：
  - （wataさん）：<https://atcoder.jp/contests/ahc030/submissions/50443474>
  - （てりーさん）：<https://atcoder.jp/contests/ahc030/submissions/50450965>
- その他
  - （bowwowforeachさん）<https://bowwowforeach.hatenablog.com/entry/2023/08/24/205427>
  - 正規分布：<https://manabitimes.jp/math/931>
  - ベイズの定理：<https://manabitimes.jp/math/804>
  - ベイズ推定：<https://manabitimes.jp/math/1390>
  - 最尤推定：<https://manabitimes.jp/math/1184>
  - 累積分布関数：<https://mathlandscape.com/distrib-func/>
  - ノルム：<https://manabitimes.jp/math/1269>
  - ノルム最小化：<https://www.msi.co.jp/solution/nuopt/docs/techniques/articles/norm-minimization.html>
  - 凸最適化，L1, L2ノルム最小化：<https://qiita.com/taka_horibe/items/9536931fbb26a6c51f6b>
  - Pythonの凸最適化ツール：<https://qiita.com/taka_horibe/items/9536931fbb26a6c51f6b>
  - フォルダ構成など（てりーさん）<https://github.com/terry-u16/ahc030>
