---
ID: 202
post_title: 【fast.ai】 Data Block API解説
author: shunkakinoki
post_excerpt: ""
layout: post
permalink: >
  https://blog.shunkakinoki.com/ja/fastai-datablock/
published: true
post_date: 2019-03-08 18:14:33
tags:
  - fast.ai
categories:
  - 人工知能
---
本記事はfast.aiのwikiの[Data Block][1]ページの要約となります。  
筆者の理解した範囲内で記載します。

# 概要

training, validation, testに用いるデータの読み込みを行うための[DataBunch][2]の設定をわずか数行のコードで行うことができる！

**細かく設定できる非常にフレキシブルなAPI**

今回は以下の7つの例を用いていきます。 1. Binary Classification 2. Multi Label Classification 3. Mask Segmentation 4. Object Detection 5. Text Language Model 6. Text Classification 7. Tabular

## 1\. [Binary Classification] 最初に、MNISTを用いた例を挙げていきます。

<pre><code class="python">from fastai.vision import *
</code></pre>

<pre><code class="python">path = untar_data(URLs.MNIST_TINY)
tfms = get_transforms(do_flip=False)
path.ls()
</code></pre>

<pre><code class="python">(path/'train').ls()
</code></pre>

### 肝心のDataBlockは下から！

<pre><code class="python">data = (ImageList.from_folder(path)     #どこからのデータか? -&gt; pathの中のフォルダとサブフォルダで、ImageList
        .split_by_folder()              #train/validをどのように分けるか? -&gt; フォルダをそのまま用いる
        .label_from_folder()            #labelをどのように付けるか? -&gt; フォルダの名前から転用する
        .add_test_folder()              #testを付け足す
        .transform(tfms, size=64)       #Data augmentationを用いるか? -&gt; size64のtfmsを用いる 
        .databunch())                   #DataBunchへと変換する
</code></pre>

DataBlockを実際に読み込んで出力してみましょう

<pre><code class="python">data.show_batch(3, figsize=(6,6), hide_axis=False)
</code></pre>

![1][3]

すげええええ、本当にたった数行でtrain/validation/testに分けてdata augmentation用いたdatabunchを作成できた！

## 2\. [Multi Label Classification] 次は、planetを用いた例を挙げていきます。

<pre><code class="python">planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
</code></pre>

<pre><code class="python">data = (ImageList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        #どこからのデータか? -&gt; planet内のtrainフォルダで、ImageList
        .random_split_by_pct()
        #train/validをどのように分けるか? -&gt; ランダムでdefaultの20%の割合でvalidへ
        .label_from_df(label_delim=' ')
        #labelをどのように付けるか? -&gt; csvファイルを用いる
        .transform(planet_tfms, size=128)
        #Data augmentationを用いるか?-&gt; size128のtfmsを用いる
        .databunch())                          
        #DataBunchへと変換する
</code></pre>

同様にDataBlockを実際に読み込んで出力してみましょう

<pre><code class="python">data.show_batch(rows=2, figsize=(9,7))
</code></pre>

![2][4]

きちんとmulti-label-classificationが読み込めていますね。

## 3\. [Mask Segmentation] camvidを用いた例を挙げていきます。

<pre><code class="python">camvid = untar_data(URLs.CAMVID_TINY)
path_lbl = camvid/'labels'
path_img = camvid/'images'
</code></pre>

<pre><code class="python">codes = np.loadtxt(camvid/'codes.txt', dtype=str); codes
</code></pre>

labelを付け足す関数を自作します。

<pre><code class="python">get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
</code></pre>

`tfm_y＝True`によって、data_augmentationが元々のmaskにも適用されるそう。

<pre><code class="python">data = (SegmentationItemList.from_folder(path_img)
        #どこからのデータか? -&gt; path_imgで、SegmentationItemList
        .random_split_by_pct()
        #train/validをどのように分けるか? -&gt; ランダムでdefaultの20%の割合でvalidへ
        .label_from_func(get_y_fn, classes=codes)
        #labelをどのように付けるか? -&gt; get_y_func
        .transform(get_transforms(), tfm_y=True, size=128)
        #Data augmentationを用いるか?-&gt; Standard transforms で tfm_y=True, size=128を指定
        .databunch())
        #DataBunchへと変換する
</code></pre>

出力します。

    data.show_batch(rows=2, figsize=(7,5))
    

![3][5]

簡単すぎ、、なのにすげええええ どんどんいきます。

## 4\. [Object Detection] cocoを用いた例を挙げます。

<pre><code class="python">coco = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco/'train.json')
img2bbox = dict(zip(images, lbl_bbox))
get_y_func = lambda o:img2bbox[o.name]
</code></pre>

<pre><code class="python">data = (ObjectItemList.from_folder(coco)
        #どこからのデータか? -&gt; cocoで、ObjectItemList
        .random_split_by_pct()                          
        #train/validをどのように分けるか? -&gt; ランダムでdefaultの20%の割合でvalidへ
        .label_from_func(get_y_func)
        #labelをどのように付けるか? -&gt; get_y_func
        .transform(get_transforms(), tfm_y=True)
        #Data augmentationを用いるか?-&gt; Standard transforms で tfm_y=True
        .databunch(bs=16, collate_fn=bb_pad_collate))   
        #DataBunchへと変換する -&gt; bb_pad_collateによって一部のbbを出力
</code></pre>

<pre><code class="python">data.show_batch(rows=2, ds_type=DatasetType.Valid, figsize=(6,6))
</code></pre>

![4][6]

細かい設定をいじって非常にフレキシブルなAPIですね

## 5\. [Text Language Model] IMDBを用いた例を挙げます。

<pre><code class="python">from fastai.text import *
</code></pre>

<pre><code class="python">imdb = untar_data(URLs.IMDB_SAMPLE)
</code></pre>

<pre><code class="python">data_lm = (TextList.from_csv(imdb, 'texts.csv', cols='text')
        #どこからのデータか? -&gt; imdb の'texts.csv'のなかの'text'column で、TextList
        .random_split_by_pct()
        #train/validをどのように分けるか? -&gt; ランダムでdefaultの20%の割合でvalidへ
        .label_for_lm()
        #labelをどのように付けるか? -&gt; Language Modelから
        .databunch())
        #DataBunchへと変換する
</code></pre>

<pre><code class="python">data_lm.show_batch()
</code></pre>

![5][7]

これだけでLanguage Modelを鍛えることができます。

### 6\. [Text Classification] IMDBを用いた例を挙げます。

上のLanguage Modelを延長して、

<pre><code class="python">data_clas = (TextList.from_csv(imdb, 'texts.csv', cols='text')
        #どこからのデータか? -&gt; imdb の'texts.csv'のなかの'text'column  で、TextList
        .split_from_df(col='is_valid')
        #train/validをどのように分けるか? -&gt; 'is_valid' column にて分割
        .label_from_df(cols='label')
        #labelをどのように付けるか? -&gt; 'label' column dfを参照する
        .databunch())
        #DataBunchへと変換する
</code></pre>

完成！

    data_clas.show_batch()
    

![6][8]

### 7\. [Text Classification] IMDBを用いた例を挙げます。

<pre><code class="python">from fastai.tabular import *
</code></pre>

<pre><code class="python">adult = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(adult/'adult.csv')
dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cont_names = ['education-num', 'hours-per-week', 'age', 'capital-loss', 'fnlwgt', 'capital-gain']
procs = [FillMissing, Categorify, Normalize]
</code></pre>

<pre><code class="python">data = (TabularList.from_df(df, path=adult, cat_names=cat_names, cont_names=cont_names, procs=procs)
        #どこからのデータか? -&gt; dfからのTabular List
        .split_by_idx(valid_idx=range(800,1000))
        #train/validをどのように分けるか? -&gt; val_idxの800から1000
        .label_from_df(cols=dep_var)
        #labelをどのように付けるか? -&gt; dep var＝ターゲットを用いる
        .databunch())
        #DataBunchへと変換する
</code></pre>

<pre><code class="python">data.show_batch()
</code></pre>

![7][9]

# 個人的な振り返り

1.  継続的にアウトプットすることでfast.aiの理解度を深め, fast.ai の掲げているdemocratizationへ寄与する。
2.  より深い部分のコードを解説していく。
3.  次回、DataBlockのより細かい設定のAPIを解説していこうと思います。

最後に  
間違いやご指摘などが御座いましたらご教示願います！

 [1]: https://docs.fast.ai/data_block.html
 [2]: https://docs.fast.ai/basic_data.html#DataBunch
 [3]: ../img/code/datablock/1.png
 [4]: ../img/code/datablock/2.png
 [5]: ../img/code/datablock/3.png
 [6]: ../img/code/datablock/4.png
 [7]: ../img/code/datablock/5.png
 [8]: ../img/code/datablock/6.png
 [9]: ../img/code/datablock/7.png