# StuCrsフレームワークのユーザーガイド

このサイトではStuCrsフレームワークの大まかな仕様、そしてユーザーが今後どのようにフレームワークを活用していくかを解説します。  
[研究のリポジトリはこちら](https://github.com/chitono/StuCrs)

## 目次

1. [StuCrsを動かす準備](#StuCrsを動かす準備)
2. [RcVariableの仕様](#RcVariableの仕様)
3. [関数の紹介](#関数の紹介)
4. [Layerの実装](#Layerの実装)
5. [Optimizerの実装](#Optimizerの実装)
6. [Modelの実装](#Modelの実装)
7. [データの用意](#データの用意)
8. [ニューラルネットワークの構築](#ニューラルネットワークの構築)
9. [学習方法](#学習方法)
10. [CNN実装](#CNN実装)
11. [CUDA対応](#CUDA対応)
12. [今後の課題](#今後の課題)

## StuCrsを動かす準備
本研究のリポジトリの中に含まれる`Dockerfile`、`compose.yaml`ファイルを用いてコンテナを立ち上げてください。(Dockerに関する使い方の説明は省略します。)  
**注意** Dockerfileの**FROM** のところは自分の環境に適したCUDAのバージョンを指定してください。また、NVIDIA製のGPUを使用しない場合、ubuntu20.04をお使いください。

<details>
  
  <summary>コード</summary>
    
```bash
docker build -t cuda-im ./ #イメージ名はcuda-im
docker compose up -d # イメージ名をcuda-imと設定しているので、イメージ名を変更したい場合はyamlファイルの中を変更してください。
```

</details>

## RcVariableの仕様

## 関数の紹介
### 四則演算
- Add
  <details>
  <summary>コード</summary>

  ```rust
  #[test]
      fn add_test() {
          use crate::core_new::ArrayDToRcVariable;
  
          let a = array![1.0, 1.0, 1.0, 1.0, 1.0].rv();
  
          let b = array![2.0, 2.0, 2.0, 2.0, 2.0].rv();
  
          let mut c = a.clone() + b.clone();
  
          println!("c = {}", c.data());
  
          c.backward(false);
  
          println!("a_grad = {:?}", a.grad().unwrap().data());
          println!("b_grad = {:?}", b.grad().unwrap().data());
      }
  ```

  </details>
  
- Sub
- Mul
- Div
- Neg
### 数学関数
- Square
- Pow
- Exp
- Sin
- Cos
- Tanh
- Log 
### 行列用関数
- Reshape
- Transpose
- Sum
- broadcast_to
- sum_to
- MatMul
### ニューラルネットワーク用関数


### 活性化関数
- Sigmoid
- Tanh
- Relu
- softmax

### 損失関数
- cross_entropy


## Layerの実装







## Optimizerの実装



### SGD (Stochastic Gradient Descent)

```rust

```

### Adam 



### AdaGrad 




## Modelの実装







## データの用意



## ニューラルネットワークの構築






## 学習方法




## CNN実装

### Convolution Operations




## CUDA対応




## 今後の課題




- 
- 
