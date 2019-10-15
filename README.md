# VLMMLR Packer Contest

[Click here to see my slides](VLMMLR_packing_contest.pdf)

## Run

#### First run
```sh
cd VLMMLR_packer_contest
python3 run.py
python3 analyze.py
```

#### Named run
```sh
cd VLMMLR_packer_contest
python3 run.py --model my_model.h5
python3 analyze.py --model my_model.h5
```

#### Using other dataset
```sh
cd VLMMLR_packer_contest
python3 run.py --training_data data/training_data.first_block.500000.csv
python3 analyze.py 
```

#### Getting more training data

I have a larger training data set [here](https://drive.google.com/file/d/1S7Q8GGTs3fMZ2POA0j8xyqMlg0I8aQ5X/view?usp=sharing
)

## Submitting Results

When you are done training a model, run `score.py --model [model name]`. Don’t do this too often. Strictly speaking you shouldn’t even have access to this ability.

Add your .h5 model file and your run.py file to `results/` and report the output of `score.py` in results.md. Push to master when you are done (may need to pull first).


## Useful links

https://keras.io
