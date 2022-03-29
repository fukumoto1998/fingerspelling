# アルゴリズム

あいうえおの人差し指の第三関節文字のみを識別する場合まず小指の曲がり具合の状態で3つのグループに分けることができる。伸びている場合は「い」が確定する。第三関節で曲がっている場合は「あ」または「う」のどちらかになり、中指と人差し指が立っている場合は「う」となり立っていない場合は「あ」となる。また、小指が第二関節で曲がっている場合は「え」または「お」のどちらかになり、人差し指と親指が合わさっている場合は「お」となり、合わさっていない場合は「え」となる。以上のことから人差し指の第三関節文字を識別することが可能となる。

以後、小指が立っているか判定する分類器を小指状態分類器、人差し指と親指が合わさっているか判定する分類器を人差し親指状態分類器、中指と人差し指が立っているか判定する分類器を中人差し指状態分類器と呼称する。



#### フローチャート

```mermaid
graph TD
	A{小指が立っているか <br> '小指状態分類器'}
	B{人差し指と親指が <br> 合わさっているか <br> '人差し親指状態分類器'}
	C{中指と人差し指が <br> 立っているか <br> '中人差し指状態分類器'}

	A --> |第二関節で曲がっている| B
	A --> | 伸びている| い
	A --> |第三関節で曲がっている| C
	B --> |Yes| お
	B --> |No| え
	C --> |Yes| う
	C --> |No| あ
```

# 各指がどの関節で曲がっているか定義したリスト

| 指文字     | 親指       | 人差し指   | 中指    | 薬指       | 小指 |　親指と人差し指と中指の状態 |
| ---------- | ---------- | ---------- | ------- | ---------- | ---- | ---- |
| あ   | 伸びている | 第三関節    | 第三関節    | 第三関節 | 第三関節    | 離れている |
| い   | 伸びていない | 第三関節    | 第三関節    | 第三関節 | 伸びている |離れている|
| う   | 伸びていない | 伸びている | 伸びている | 第三関節 | 第三関節    |離れている|
| え   | 第二関節 | 第二関節 | 第二関節 | 第二関節 | 第二関節 |離れている|
| お   | 第二関節 | 第二関節 | 第二関節 | 第二関節 | 第二関節 |合わさっている|

※　親指の伸びているの定義はグッドのように指立っている時を伸びているとしている



# 学習

## 最も精度が高い特徴量リスト

### 小指状態分類器

##### 角度補正後のデータ

- 極座標
  
  - 原点と薬指の第三関節
- 3点間の角度
  - 小指の第三関節, 小指の第二関節, 小指の第一関節
  - 小指の第三関節, 小指の第一関節, 小指の指先
  - 小指の第二関節, 小指の指先, 小指の第三関節
  - 小指の第二関節, 小指の指先, 小指の第一関節
  - 小指の指先, 小指の第一関節, 小指の第二関節
- 2点間の角度
  - 小指の第三関節と小指の指先



### 中人差し指状態分類器

##### 角度補正後のデータ
- 2点間の距離
  - 原点と中指の指先
  



### 親指状態分類器

##### 角度補正後のデータ

- 極座標

  - 親指の第一関節と親指の指先
- 3点間の角度
  - 親指の第一関節,中指の指先,親指の指先
  - 親指の指先,親指の第一関節,中指の指先
  - 人差し指の指先,中指の指先,親指の指先
- 2点間の距離
  - 原点と中指の指先



## 学習パラメータ

### 決定木のパラメータ

| name         | parameter |
| ------------ | --------- |
| max_depth    | 5         |
| random_state | 69        |



## 学習結果

### 小指状態分類器

「あ」、「い」、「う」、「え」、「お」のデータで学習

- train data size：119
- test data size   : 65

![ダウンロード](https://user-images.githubusercontent.com/87710914/160513297-0c77e091-65bf-430f-8321-c462a1738a95.png)

- 横軸：世代数、縦軸：各世代の最大正答率

#### 混同行列

![img](data:image/png;base64,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)



| classes | label  |
| ------- | ------ |
| 0       | あ、う |
| 1       | い     |
| 2       | え、お |



### 中人差し指状態分類器

「あ」、「う」のみのデータで学習

- train data size：48
- test data size   : 26

![img](data:image/png;base64,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)

- 横軸：世代数、縦軸：各世代の最大正答率

#### 混同行列

![img](data:image/png;base64,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)

| classes | label |
| ------- | ----- |
| 0       | あ    |
| 1       | う    |



### 親指状態分類器

「え」、「お」のみのデータで学習

- train data size：47
- test data size   : 26

![img](data:image/png;base64,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)

- 横軸：世代数、縦軸：各世代の最大正答率

![img](data:image/png;base64,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)

| classes | label |
| ------- | ----- |
| 0       | え    |
| 1       | お    |
