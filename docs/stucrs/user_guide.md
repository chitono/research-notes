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


    
```bash
docker build -t cuda-im ./ #イメージ名はcuda-im
docker compose up -d # イメージ名をcuda-imと設定しているので、イメージ名を変更したい場合はyamlファイルの中を変更してください。
```



## RcVariableの仕様

RcVariableはこのフレームワーク独自の構造体であり、値や微分の値を保持するVariabel構造体をRc<RefCell<>>型で包んだものです。Rc、RefCell型により、変数Variableを可変として共同所有することができます。また、この構造体に演算子の関数や、exp、sinといった関数、さらに行列を扱う関数を実装することで、自動で行列の微分の値を計算する設計となっています。RcVaribleの自動設計による機能により、開発が非常にスムーズになります。  
  
例えば **y = sin(x)** の場合を考えてみましょう。
```rust 
fn sin_test() {
        use crate::core_new::ArrayDToRcVariable;

        let x = array![3.0, 3.0, 3.0].rv(); //xの値

        let mut y = sin(&x); //yをsinとxを用いて定義

        println!("y = {}", y.data()); // 0.1411 yの値が求まる

        y.backward(false); // backward関数を呼び出す

        println!("x_grad = {:?}", x.grad().unwrap().data()); // -0.9899 xの微分の値
    }

```

この時 sin関数の微分は **cos(x)** なので、 **x** の値の3.0をcosに代入した値が、 **x=3.0** の時のyをxで微分した値です。このとき、微分の値は-0.9899と求められますが、この値を求めるのに必要なコードはyの式をsinとaを用いて定義し、yのbackward関数を呼び出すだけです。これだけで微分できるのは、RcVariableが裏側で、微分の式を自動で構築してくれるからです。

  
微分を自動化しているため、下の **sigmoid関数** のような複雑な関数も自動で求めることができます。

```rust 
pub fn sigmoid_simple(x: &RcVariable) -> RcVariable {
    let mainasu_x = -x.clone();
    let y = 1.0f32.rv() / (1.0f32.rv() + exp(&mainasu_x));
    y
}
```

この自動微分は多変数関数の微分、すなわち偏微分にも対応しています。例えば二つの変数a,bをかけた値cの式は **c = a×b** となりますが、この場合、∂c/∂a、∂c/∂bを求めることができます。

```rust 

fn mul_test() {
　　use crate::core_new::ArrayDToRcVariable;

　　let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

　　let b = array![2.0, 2.0, 2.0, 2.0, 2.0].rv();

　　let mut c = (a.clone() * b.clone());

　　println!("c = {}", c.data()); // 5.0

　　c.backward(false);

　　println!("a_grad = {:?}", a.grad().unwrap().data()); // 2.0
　　println!("b_grad = {:?}", b.grad().unwrap().data()); // 3.0
}

```


## 関数の紹介



### 四則演算
- Add
  

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

- Dense

```rust

let mut model = BaseModel::new();
model.stack(L::Dense::new(1000, true, None, Activation::Sigmoid));

```

- Linear

- ActivationLayer

- Conv2d

```rust

let out_channels = 4;
let kernel_size = (3, 3);
let stride_size = (1, 1);
let pad_size = (0, 0);
let biased = false;
let mut model = BaseModel::new();
model.stack(L::Conv2d::new(out_channels, kernel_size, stride_size, pad_size, biased));

```

- Maxpool2d

- Dropout

- Flatten




## Optimizerの実装






### SGD (Stochastic Gradient Descent)

```rust

let mut model = BaseModel::new();
let mut optimizer = SGD::new(lr);
optimizer.setup(&model);

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
