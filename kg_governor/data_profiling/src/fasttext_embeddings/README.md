Here goes the embedding file for fastText: cc.en.50.bin.


To generate it:

1. Download the 300-dimensional model using:

```
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
```

2. Reduce the dimension to 50 using:

```
import fasttext
import fasttext.util
ft = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(ft, 50)
```
3. Save the model using:
```
ft.save_model('cc.en.50.bin')
```