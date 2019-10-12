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

#### USing other dataset
```sh
cd VLMMLR_packer_contest
python3 run.py --training_data data/training_data.first_block.500000.csv
python3 analyze.py 
```

## Options

The master branch has 35 input values: 15 rates + 20 one-hot encoded AA data.
Instead, you can choose to have just the 15 rate values by using the `JackMaguire/NoAAInfo` branch as your starting point
(`git checkout JackMaguire/NoAAInfo; git checkout -b my_new_branch_name`).

Another option is to split the 35 values into two separate input layers.
`JackMaguire/split_input` inputs the 15 rate values to `input1` and the
20 one-hot encodings into `layer2`.

Both branches handle all of this logic for you so you shouldn't even notice a difference.

## Useful links

https://keras.io
