---
title: "StuCrs | 深層学習の原理探究へ向けたRust製フレームワーク"
---

## 研究概要

本研究では Rust言語を用いて「StuCrs」というディープラーニングのフレームワークを一から実装、開発しました。StuCrsというフレームワークの特徴はフルRust実装で、直感的に原理を理解できるシンプルな構造となっており、
ユーザーが一から実装し、深層学習の原理の理解を深めてもらう教材としての役割を果たすフレームワークです。また、Rust言語を学びたい方にとっても良いサンプルコードです。

## 背景
・TensorFlowやPyTorchといった既存のフレームワークのほとんどがドキュメントやコミュニティが英語だったりと、日本語によるフレームワークの開発にとって障壁となっている。

・機械学習の開発がpythonやC系言語に比べてRustは遅れている。


## 研究のコンセプト
・日本語によってコードの説明をすることでユーザー自らが一からフレームワークを実装してもらい、深層学習の原理を探究してもらうこと。

・日本語のコミュニティを構築しやすい国産のフレームワークを、機械学習で開発途上のRustで実装することで、さらなる日本でのRustにおける深層学習のコミュニティを活発にし、開発を促すこと。

## 研究にあたって
本研究は下の著書『ゼロから作るDeep Learning③フレームワーク編』をもとにして実装しています。著者である斎藤康毅氏に著書の考えや表現の使用を許可していただいたことに感謝を申し上げるとともに、この著書オリジナルのフレームワークDeZeroも研究の参考として利用させていただいています。



## ニュース
本研究は第19回高校生理科研究発表会に出場し、最優秀賞をいただきました。参考にさせていただいた研究の方、審査していただいた方々に感謝申し上げます。大会に提出したポスターはassetsフォルダーでみることができます。


## ユーザーガイド
StuCrsフレームワークの仕様や使い方、活用方法などを解説しているサイトはこちら
[ユーザーガイド](user_guide.md)

## ドキュメント


開発した深層学習のフレームワーク「StuCrs」の実装までのコードの説明をこちらのドキュメントで見ることができます。これを読んでぜひ一からRustでフレームワークを実装してみましょう！
<https://docs.google.com/document/d/1jJL_ijYnqIFADSTfTqLcnNre754g24bE963L_r3hwus/edit?usp=sharing>


## 使用した外部のクレート

本研究で必要とする外部クレートとバージョンは下記の通りです。

- [ndarray-0.16.0](https://docs.rs/ndarray/0.16.0/ndarray/index.html)
- [ndarray_stats-0.6.0](https://docs.rs/ndarray-stats/0.6.0/ndarray_stats/index.html)
- [ndarray-rand-0.15.0](https://docs.rs/ndarray-rand/latest/ndarray_rand/index.html)
- [mnist-0.6.0](https://docs.rs/mnist/latest/mnist/index.html)
- [rand-0.8](https://docs.rs/rand/latest/rand/index.html)
- [rand_distr-0.4.0](https://docs.rs/rand_distr/0.4.0/rand_distr/index.html)
- [fxhash-0.2.1](https://docs.rs/fxhash/latest/fxhash/index.html)



NVIDIAのGPUで実行できる機能も提供しています。その場合はstucrs-gpuをダウンロードし、また下記のtensor_frameクレートを使用します。

- [tensor_frame](https://docs.rs/tensor_frame/latest/tensor_frame/index.html) （オプション）

## 本研究のリポジトリはこちら
<https://github.com/chitono/StuCrs>
